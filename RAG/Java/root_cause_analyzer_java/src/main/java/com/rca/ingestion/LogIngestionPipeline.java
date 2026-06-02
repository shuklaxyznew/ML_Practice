package com.rca.ingestion;

import com.rca.model.ProcessedLogs;
import org.springframework.stereotype.Service;

import java.util.*;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

/**
 * Data Ingestion Layer — filters noise, masks PII, summarizes logs.
 * Pro tip from spec: Never send 10,000 lines raw to the LLM.
 */
@Service
public class LogIngestionPipeline {

    private static final int MAX_SUMMARY_LINES = 30;

    // PII masking patterns
    private static final List<String[]> PII_PATTERNS = List.of(
        new String[]{"\\b\\d{3}-\\d{2}-\\d{4}\\b",                          "[SSN]"},
        new String[]{"\\b[A-Za-z0-9._%+\\-]+@[A-Za-z0-9.\\-]+\\.[A-Za-z]{2,}\\b", "[EMAIL]"},
        new String[]{"\\b(?:\\d{4}[- ]?){3}\\d{4}\\b",                      "[CARD]"},
        new String[]{"\\b(?:\\d{1,3}\\.){3}\\d{1,3}\\b",                    "[IP]"},
        new String[]{"(?i)password[=:]\\S+",                                  "password=[MASKED]"},
        new String[]{"(?i)token[=:]\\S+",                                     "token=[MASKED]"},
        new String[]{"(?i)api[_\\-]?key[=:]\\S+",                            "api_key=[MASKED]"}
    );

    private static final Pattern CRITICAL_PATTERN =
        Pattern.compile("\\b(ERROR|CRITICAL|EXCEPTION|FATAL|SEVERE)\\b", Pattern.CASE_INSENSITIVE);

    private static final Pattern WARNING_PATTERN =
        Pattern.compile("\\b(WARN|WARNING)\\b", Pattern.CASE_INSENSITIVE);

    public ProcessedLogs process(String rawLogs, String serviceName) {
        List<String> lines = Arrays.stream(rawLogs.split("\\r?\\n"))
            .map(String::trim)
            .filter(l -> !l.isEmpty())
            .map(this::maskPii)
            .collect(Collectors.toList());

        List<String> errorLines = lines.stream()
            .filter(l -> CRITICAL_PATTERN.matcher(l).find())
            .collect(Collectors.toList());

        List<String> warningLines = lines.stream()
            .filter(l -> WARNING_PATTERN.matcher(l).find())
            .collect(Collectors.toList());

        List<String> summaryLines = buildSummary(errorLines, warningLines, lines);

        return ProcessedLogs.builder()
            .rawLines(lines)
            .errorLines(errorLines)
            .warningLines(warningLines)
            .summary(String.join("\n", summaryLines))
            .errorCount(errorLines.size())
            .warningCount(warningLines.size())
            .serviceName(serviceName)
            .build();
    }

    private String maskPii(String line) {
        for (String[] entry : PII_PATTERNS) {
            line = line.replaceAll(entry[0], entry[1]);
        }
        return line;
    }

    /**
     * Priority ordering: ERRORs → WARNs → first/last INFO lines.
     * Capped at MAX_SUMMARY_LINES to control LLM token cost.
     */
    private List<String> buildSummary(List<String> errors, List<String> warnings, List<String> all) {
        Set<String> seen = new LinkedHashSet<>();
        List<String> result = new ArrayList<>();

        for (String line : errors) {
            if (seen.add(line)) {
                result.add(line);
                if (result.size() >= MAX_SUMMARY_LINES) return result;
            }
        }
        for (String line : warnings) {
            if (seen.add(line)) {
                result.add(line);
                if (result.size() >= MAX_SUMMARY_LINES) return result;
            }
        }

        // Pad with first 5 + last 5 info lines for context
        List<String> context = new ArrayList<>();
        if (all.size() > 0) context.addAll(all.subList(0, Math.min(5, all.size())));
        if (all.size() > 5) context.addAll(all.subList(Math.max(5, all.size() - 5), all.size()));

        for (String line : context) {
            if (seen.add(line)) {
                result.add(line);
                if (result.size() >= MAX_SUMMARY_LINES) break;
            }
        }

        return result;
    }
}
