package com.rca.model;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.List;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class Incident {

    private String title;
    private String description;
    private String resolution;
    private List<String> keywords;
    private double similarityScore;
}
