package com.example;

import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.splitter.DocumentByParagraphSplitter;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.ollama.OllamaChatModel;
import dev.langchain4j.model.ollama.OllamaEmbeddingModel;
import dev.langchain4j.model.openai.OpenAiEmbeddingModel;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.EmbeddingStoreIngestor;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;
import lombok.Data;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.Primary;

import java.time.Duration;

@Configuration
@ConfigurationProperties(value = "ai.embedding")
@Data
public class AppConfig {
    String url;

    String modelName;

    String googleApiKey;

    @Bean
    @Qualifier("openAI")
    EmbeddingModel openAI() {
        return OpenAiEmbeddingModel.builder()
                .baseUrl(url)
                .apiKey("dummy")
                .modelName(modelName)
                .build();
    }

    @Bean
    @Qualifier("ollama")
    EmbeddingModel ollama() {
        return OllamaEmbeddingModel.builder().baseUrl(url).modelName(modelName).build();

    }

    @Bean
    @Qualifier("llama3")
    @Primary
    ChatLanguageModel llama3() {
        ChatLanguageModel llm = OllamaChatModel.builder()
                .baseUrl(url)
                .modelName("llama3")
                .timeout(Duration.ofMinutes(3))
                .temperature(0.3)
                .build();
        return llm;
    }
    @Bean
    @Qualifier("llama3.1")
    ChatLanguageModel llama3_1() {
        ChatLanguageModel llm = OllamaChatModel.builder()
                .baseUrl(url)
                .modelName("llama3.1")
                .timeout(Duration.ofMinutes(3))
                .build();
        return llm;
    }

    @Bean
    @Qualifier("mistral")
    ChatLanguageModel mistral() {
        ChatLanguageModel mistral = OllamaChatModel.builder()
                .baseUrl(url)
                .modelName("mistral:7b")
                .timeout(Duration.ofMinutes(3))
                .build();
        return mistral;
    }

    @Bean
    @Qualifier("gemini")
    ChatLanguageModel google() {
        ChatLanguageModel chatModel = GoogleAiGeminiChatModel.builder()
                // Use a model ID suitable for chat, such as the fast Flash model
                .modelName("gemini-2.5-flash")
                .apiKey(googleApiKey)
                .timeout(Duration.ofMinutes(2))
                // Optional configuration:
                .temperature(0.3) // Controls randomness (0.0 to 1.0)
                .maxOutputTokens(2048) // Maximum number of tokens in the response
                .build();
        return chatModel;
    }

    @Bean
    EmbeddingStore<TextSegment> store() {
        return new InMemoryEmbeddingStore<>();
    }

    @Bean
    EmbeddingStoreIngestor ingestor(EmbeddingModel ollama, EmbeddingStore<TextSegment> store) {
        DocumentSplitter paragraphSplitter = new DocumentByParagraphSplitter(500, 50);
        EmbeddingStoreIngestor ingestor = EmbeddingStoreIngestor.builder()
                .documentSplitter(paragraphSplitter)
                .embeddingModel(ollama)
                .embeddingStore(store)
                .build();
        return ingestor;
    }

}
