package com.example;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.EmbeddingStoreIngestor;
import lombok.RequiredArgsConstructor;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.web.bind.annotation.*;

import java.util.stream.Collectors;

@RestController
@RequiredArgsConstructor
public class EmbeddingController {
    record EmbeddingRequest(String id, String text) {}
    record SearchRequest(String query, int k) {}

    @Qualifier("ollama")
    private final EmbeddingModel ollama;

    private final EmbeddingStore<TextSegment> store;
    private final EmbeddingStoreIngestor ingestor;
    private final ChatLanguageModel chatModel;

    @PostMapping("/embed")
    String embed(@RequestBody EmbeddingRequest request) {
        ingestor.ingest(Document.document(request.text()));
        return "ok";
    }

    @PostMapping("/search")
    String search(@RequestBody SearchRequest request) {
        var matches = store.findRelevant(
                ollama.embed(request.query()).content(),
                Math.max(request.k(), 3)
        );

        String context = matches.stream()
                .map(m -> m.embedded().text())
                .collect(Collectors.joining("\n\n"));

        return chatModel.generate(String.format(
                "Use the following context to answer the question.\nContext:\n%s\n\nQuestion: %s",
                context, request.query()
        ));
    }
}
