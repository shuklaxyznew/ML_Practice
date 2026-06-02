package com.rca.model;

import lombok.Builder;
import lombok.Data;

import java.util.List;

@Data
@Builder
public class AnalyzeResponse {

    private String rootCause;
    private int confidence;
    private String severity;
    private int errorCount;
    private int warningCount;
    private List<String> affectedServices;
    private Integer mttrEstimateMinutes;
    private List<Evidence> evidence;
    private List<ActionItem> suggestedActions;
    private List<SimilarIncident> similarPastIncidents;

    @Data
    @Builder
    public static class Evidence {
        private String text;
        private String type; // "error" | "warn"
    }

    @Data
    @Builder
    public static class ActionItem {
        private int step;
        private String priority; // "immediate" | "short-term" | "long-term"
        private String title;
        private String detail;
    }

    @Data
    @Builder
    public static class SimilarIncident {
        private String title;
        private String resolution;
        private double similarity;
    }
}
