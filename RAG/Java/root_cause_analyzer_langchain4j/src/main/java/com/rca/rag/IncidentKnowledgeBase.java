package com.rca.rag;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.Metadata;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.EmbeddingStoreIngestor;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

/**
 * Allows runtime ingestion of new post-mortems / JIRA tickets into the
 * LangChain4j EmbeddingStore — equivalent to Python's VectorStore.add_documents().
 *
 * The RetrievalAugmentor in LangChain4jConfig automatically queries this
 * store on every SreAssistant.analyze() call, so newly added incidents are
 * immediately available for retrieval.
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class IncidentKnowledgeBase {

    private final EmbeddingStore<TextSegment> embeddingStore;
    private final EmbeddingModel              embeddingModel;

    /**
     * Embed and store a new incident.
     * Equivalent to Python: vector_store.add_documents([Document(...)])
     */
    public void addIncident(String title, String description, String resolution) {
        String text = String.format("Title: %s\nDescription: %s\nResolution: %s",
            title, description, resolution);

        Document doc = Document.from(text, Metadata.from("title", title));

        EmbeddingStoreIngestor.builder()
            .embeddingModel(embeddingModel)
            .embeddingStore(embeddingStore)
            .build()
            .ingest(doc);

        log.info("Ingested new incident into EmbeddingStore: '{}'", title);
    }
}
