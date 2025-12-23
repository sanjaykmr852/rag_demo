package com.example.config;

import dev.langchain4j.store.embedding.pgvector.PgVectorEmbeddingStore;
import net.sourceforge.tess4j.Tesseract;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import java.sql.SQLException;

@Configuration
public class DocumentConfig {

    @Bean
    public Tesseract tesseract() {
        Tesseract tesseract = new Tesseract();
        tesseract.setDatapath("D:\\GPT\\tessaract"); // Adjust path as needed
        return tesseract;
    }

    @Bean
    public PgVectorEmbeddingStore pgVectorStore() throws SQLException {
      /*  Connection connection = DriverManager.getConnection(
                "jdbc:postgresql://localhost:5432/postgres",
                "postgres",
                "postgres"
        );*/
        PgVectorEmbeddingStore store = PgVectorEmbeddingStore.builder().host("localhost").
                port(5432).database("postgres").user("postgres").password("postgres")
                .createTable(false).dropTableFirst(false)
                .table("ai.docs").dimension(768)
                .build();
        return store;
    }
}
