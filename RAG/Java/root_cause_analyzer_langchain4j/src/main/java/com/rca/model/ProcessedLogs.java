package com.rca.model;

import lombok.Builder;
import lombok.Data;

import java.util.List;

@Data
@Builder
public class ProcessedLogs {
    private List<String> rawLines;
    private List<String> errorLines;
    private List<String> warningLines;
    private String       summary;
    private int          errorCount;
    private int          warningCount;
    private String       serviceName;
}
