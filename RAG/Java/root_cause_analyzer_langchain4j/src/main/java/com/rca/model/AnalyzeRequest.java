package com.rca.model;

import jakarta.validation.constraints.NotBlank;
import lombok.Data;

@Data
public class AnalyzeRequest {

    @NotBlank(message = "Logs must not be empty")
    private String logs;

    private String metrics;

    private String serviceName = "unknown-service";

    private int timeWindowMinutes = 5;
}
