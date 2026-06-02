package com.rca.ingestion;

import com.rca.model.ProcessedLogs;
import org.springframework.stereotype.Service;

import java.util.*;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

/**
 * Data Ingestion Layer:
 *  1. Splits raw log string into lines
 *  2. Masks PII (emails, IPs, passwords, API keys, credit cards)
 *  3. Classifies lines by severity (ERROR / WARN)
 *  4. Builds a capped summary (max 30 lines) to control LLM token cost
 *
 * Pro tip from spec: Don't send 10,000 lines of logs to the LLM.
 * Focus on ERROR, CRITICAL, and EXCEPTION tags.
 */
@Service
public class LogIngestionPipeline {

    private static final int MAX_SUMMARY_LINES = 30;

    private static final List<String[]> PII_PATTERNS = List.of(
        new String[]{"\\b\\d{3}-\\d{2}-\\d{4}\\b",                                      "[SSN]"},
        new String[]{"\\b[A-Za-z0-9._%+\\-]+@[A-Za-z0-9.\\-]+\\.[A-Za-z]{2,}\\b",      "[EMAIL]"},
        new String[]{"\\b(?:\\d{4}[- ]?){3}\\d{4}\\b",                                   "[CARD]"},
        new String[]{"\\b(?:\\d{1,3}\\.){3}\\d{1,3}\\b",                                 "[IP]"},
        new String[]{"(?i)password[=:]\\S+",                                               "password=[MASKED]"},
        new String[]{"(?i)token[=:]\\S+",                                                  "token=[MASKED]"},
        new String[]{"(?i)api[_\\-]?key[=:]\\S+",                                         "api_key=[MASKED]"},
        new String[]{"(?i)secret[=:]\\S+",                                                 "secret=[MASKED]"}
    );

    private static final Pattern CRITICAL =
        Pattern.compile("\\b(ERROR|CRITICAL|EXCEPTION|FATAL|SEVERE)\\b", Pattern.CASE_INSENSITIVE);
    private static final Pattern WARNING =
        Pattern.compile("\\b(WARN|WARNING)\\b", Pattern.CASE_INSENSITIVE);

    public ProcessedLogs process(String rawLogs, String serviceName) {
        List<String> lines = Arrays.stream(rawLogs.split("\\r?\\n"))
            .map(String::trim)
            .filter(l -> !l.isBlank())
            .map(this::maskPii)
            .collect(Collectors.toList());

        List<String> errorLines   = lines.stream().filter(l -> CRITICAL.matcher(l).find()).toList();
        List<String> warningLines = lines.stream().filter(l -> WARNING.matcher(l).find()).toList();
        List<String> summary      = buildSummary(errorLines, warningLines, lines);

        return ProcessedLogs.builder()
            .rawLines(lines)
            .errorLines(errorLines)
            .warningLines(warningLines)
            .summary(String.join("\n", summary))
            .errorCount(errorLines.size())
            .warningCount(warningLines.size())
            .serviceName(serviceName)
            .build();
    }

    private String maskPii(String line) {
        for (String[] rule : PII_PATTERNS) {
            line = line.replaceAll(rule[0], rule[1]);
        }
        return line;
    }

    private List<String> buildSummary(List<String> errors, List<String> warnings, List<String> all) {
        LinkedHashSet<String> seen = new LinkedHashSet<>();
        errors.forEach(seen::add);
        warnings.forEach(seen::add);

        // Pad with first/last 5 lines for context
        List<String> ctx = new ArrayList<>();
        ctx.addAll(all.subList(0, Math.min(5, all.size())));
        if (all.size() > 5)
            ctx.addAll(all.subList(Math.max(5, all.size() - 5), all.size()));
        ctx.forEach(seen::add);

        return seen.stream().limit(MAX_SUMMARY_LINES).collect(Collectors.toList());
    }
}
