package com.example.embedding;


import dev.langchain4j.data.document.Metadata;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.store.embedding.EmbeddingSearchRequest;
import dev.langchain4j.store.embedding.EmbeddingSearchResult;
import dev.langchain4j.store.embedding.filter.Filter;
import dev.langchain4j.store.embedding.filter.MetadataFilterBuilder;
import dev.langchain4j.store.embedding.pgvector.PgVectorEmbeddingStore;
import jakarta.servlet.http.HttpServletRequest;
import lombok.extern.slf4j.Slf4j;
import net.sourceforge.tess4j.Tesseract;
import net.sourceforge.tess4j.TesseractException;
import org.apache.commons.lang3.ObjectUtils;
import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.rendering.ImageType;
import org.apache.pdfbox.rendering.PDFRenderer;
import org.apache.pdfbox.text.PDFTextStripper;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.stream.Collectors;

@RestController
@RequestMapping("/api/documents")
@Slf4j
public class DocumentController {

    private final EmbeddingModel embeddingModel;
    private final PgVectorEmbeddingStore embeddingStore;
    private final Tesseract tesseract;
    private final ChatLanguageModel chatModel;
    // chunk configuration
    private static final int CHUNK_SIZE = 1000;      // characters per chunk
    private static final int CHUNK_OVERLAP = 200;    // overlap between chunks

    @Autowired
    public DocumentController(@Qualifier("ollama") EmbeddingModel embeddingModel,
                              PgVectorEmbeddingStore embeddingStore,
                              Tesseract tesseract, @Qualifier("gemini") ChatLanguageModel chatModel) {
        this.embeddingModel = embeddingModel;
        this.embeddingStore = embeddingStore;
        this.tesseract = tesseract;
        this.chatModel = chatModel;
    }

    @PostMapping(value = "/upload", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public String uploadDocument(@RequestParam("file") MultipartFile file,HttpServletRequest request) throws IOException {
        String content = switch (file.getContentType()) {
            case "application/pdf" -> extractPdfContent(file);
            case "text/plain" -> extractTextContent(file);
            case "image/jpeg", "image/png", "image/jpg" -> extractImageContent(file);
            default -> throw new IllegalArgumentException("Unsupported file type: " + file.getContentType());
        };

        if (content == null || content.isBlank()) {
            throw new IllegalArgumentException("No text extracted from the uploaded file");
        }
        String metadata=request.getHeader("metadata");
        List<String> chunks = chunkText(content, CHUNK_SIZE, CHUNK_OVERLAP);
        for (int i = 0; i < chunks.size(); i++) {
            Map<String,Object> metaMap=new HashMap<>();
            String chunk = chunks.get(i);

            metaMap.put("chunkIndex", String.valueOf(i));
            metaMap.put("fileName", file.getResource().getFilename());
            if(!ObjectUtils.isEmpty(metadata)){
                String tags=Arrays.stream(metadata.split(",")).collect(Collectors.joining(","));
                metaMap.put("tags",tags);
            }
            Metadata requestMeta = new Metadata(metaMap);
            TextSegment docChunk = TextSegment.from(chunk, requestMeta);
            var embedding = embeddingModel.embed(chunk).content();
            embeddingStore.add(embedding, docChunk);
        }
        return "Document processed and stored as " + chunks.size() + " chunk(s)";
    }

    private String extractPdfContent(MultipartFile file) throws IOException {
        StringBuilder combined = new StringBuilder();

        try (InputStream is = file.getInputStream();
             PDDocument pdf = PDDocument.load(is)) {

            // 1) Extract selectable text
            PDFTextStripper stripper = new PDFTextStripper();
            String pdfText = stripper.getText(pdf);
            if (pdfText != null && !pdfText.isBlank()) {
                combined.append(pdfText.trim());
                combined.append(System.lineSeparator()).append("----OCR below----").append(System.lineSeparator());
            }

            // 2) Render each page and run OCR
            PDFRenderer renderer = new PDFRenderer(pdf);
            int pages = pdf.getNumberOfPages();
            for (int i = 0; i < pages; i++) {
                BufferedImage pageImage = renderer.renderImageWithDPI(i, 300, ImageType.RGB);
                String ocrText;
                try {
                    ocrText = tesseract.doOCR(pageImage);
                } catch (TesseractException e) {
                    throw new IOException("OCR failed for PDF page " + i, e);
                }
                if (ocrText != null && !ocrText.isBlank()) {
                    combined.append(ocrText.trim());
                    if (i < pages - 1) combined.append(System.lineSeparator());
                }
            }
        }

        return combined.toString();
    }

    private String extractTextContent(MultipartFile file) throws IOException {
        byte[] bytes = file.getBytes();
        return new String(bytes, StandardCharsets.UTF_8);
    }

    private String extractImageContent(MultipartFile file) throws IOException {
        try {
            BufferedImage image = ImageIO.read(file.getInputStream());
            return tesseract.doOCR(image);
        } catch (TesseractException e) {
            throw new IOException("OCR failed", e);
        }
    }

    private List<String> chunkText(String text, int maxChars, int overlap) {
        List<String> chunks = new ArrayList<>();
        if (text == null || text.isBlank()) return chunks;

        int start = 0;
        int length = text.length();
        while (start < length) {
            int end = Math.min(start + maxChars, length);
            String chunk = text.substring(start, end).trim();
            if (!chunk.isEmpty()) chunks.add(chunk);
            if (end == length) break;
            start = Math.max(0, end - overlap);
        }
        return chunks;
    }


    @GetMapping("/search")
    public String search(
            @RequestParam("q") String query,
            @RequestParam(value = "k", defaultValue = "20") int k, HttpServletRequest httpRequest) {
        var embedding = embeddingModel.embed(query).content();
        String metadata=httpRequest.getHeader("metadata");
        Filter filter = MetadataFilterBuilder.metadataKey("tags").isIn(metadata.split(","));

        var request = EmbeddingSearchRequest.builder().queryEmbedding(embedding)
                .maxResults(k)
                .filter(filter)
                .build();
        EmbeddingSearchResult<TextSegment> matches = embeddingStore.search(request);
        List<String> results = new ArrayList<>();
        Map<String,String> resultMap=new LinkedHashMap<>();
        matches.matches().stream()
//                .filter(embeddingMatch -> embeddingMatch.score() > 0.74)
                .map(match -> match.embedded()).forEach(embedded ->
                {
                    String fileName = embedded.metadata().get("fileName");
                    String chunkIndex = embedded.metadata().get("chunkIndex");
                    String text = embedded.text();
                    String metaData = String.format("[doc=%s chunk=%s]", fileName, chunkIndex);
                    String context = metaData + "\n" + text;
                    results.add(context);
                    String existingValue=resultMap.getOrDefault(fileName,"");
                    String newValue=existingValue+"\n"+context;
                    resultMap.put(fileName,newValue);
                });
        log.info("[Referred Docs = {}]",resultMap.keySet().stream().collect(Collectors.joining(",")));
        String response = chatModel.generate(String.format("""
                You are a precise and factual assistant. 
                Use only the information provided in the context below to answer the question.
                
                --------------------
                Context:
                %s
                --------------------
                
                Question:
                %s
                
                --------------------
                Instructions:
                1. Answer concisely and accurately based only on the given context.
                2. If the answer is found, include the document name(s) or metadata you used.
                3. If the context does not contain the answer, respond exactly as:
                   "I reviewed <N> documents, but none contained the answer. Document name(s): <document name(s)"
                4. Do not make assumptions or invent information.
                --------------------
                """, resultMap.values(), query));
        return response;
    }
}