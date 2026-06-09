# RAG Pipeline — Arquitetura Completa

## Pipeline de Ingestão

```mermaid
flowchart TD
    A([Cloud Storage\nPDFs e TXTs])

    A --> B

    subgraph CHUNK["Chunking Strategy"]
        B[Text Splitter\n512 tokens · 50 de overlap]
    end

    B --> C
    B --> E

    C[Vertex AI Embeddings\ntextembedding-gecko]

    C --> D
    E --> F

    D[(Pinecone\nVetores semânticos)]
    E[(BM25 Index\nÍndice lexical — termos exatos)]

    style CHUNK fill:#1a3a2a,stroke:#2ecc71,color:#fff
    style D fill:#4a235a,stroke:#9b59b6,color:#fff
    style E fill:#1a2a3a,stroke:#3498db,color:#fff
    style A fill:#3a3a3a,stroke:#aaa,color:#fff
    style C fill:#1a3a2a,stroke:#2ecc71,color:#fff
```

---

## Pipeline de Consulta (RAG)

```mermaid
flowchart TD
    A([Pergunta do usuário\nFastAPI endpoint])

    A --> B
    A --> C

    B[Embedding da query\ntextembedding-gecko]
    C[Tokenização BM25\nMesma query em texto]

    B --> D
    C --> E

    D[Busca Semântica\nPinecone top-k]
    E[Busca Lexical\nBM25 top-k]

    D --> F
    E --> F

    subgraph HYBRID["Hybrid Search"]
        F[Fusão de resultados\nRRF — Reciprocal Rank Fusion]
    end

    F --> G

    subgraph RERANK["Reranker"]
        G[Vertex AI Rank API\nReordena chunks por relevância real]
    end

    G --> H

    H[LLM + contexto\nGemini via Vertex AI]

    H --> I

    I([Resposta contextualizada\nCom fontes e scores])

    style HYBRID fill:#1a2a3a,stroke:#3498db,color:#fff
    style RERANK fill:#3a1a1a,stroke:#e74c3c,color:#fff
    style A fill:#3a3a3a,stroke:#aaa,color:#fff
    style I fill:#3a3a3a,stroke:#aaa,color:#fff
    style B fill:#1a3a2a,stroke:#2ecc71,color:#fff
    style C fill:#1a2a3a,stroke:#3498db,color:#fff
    style D fill:#4a235a,stroke:#9b59b6,color:#fff
    style E fill:#1a2a3a,stroke:#3498db,color:#fff
    style H fill:#1a3a2a,stroke:#2ecc71,color:#fff
```

---

## Componentes — Responsabilidade de cada bloco

| Bloco | Tecnologia | Por quê |
|---|---|---|
| Cloud Storage | GCP Cloud Storage | Fonte dos documentos brutos (PDF, TXT) |
| Text Splitter | LangChain `RecursiveCharacterTextSplitter` | Chunking com 512 tokens e 50 de overlap para preservar contexto entre chunks |
| Vertex AI Embeddings | `textembedding-gecko` | Gera vetores densos para busca semântica |
| Pinecone | Vector DB gerenciado | Armazena e consulta vetores por similaridade coseno |
| BM25 Index | `rank_bm25` (Python) | Índice esparso para busca lexical — captura nomes próprios e termos exatos |
| Hybrid Fusion | RRF (Reciprocal Rank Fusion) | Combina rankings semântico + lexical sem precisar de pesos manuais |
| Reranker | Vertex AI Rank API | Reordena os top-k fundidos por relevância real em relação à query |
| LLM | Gemini via Vertex AI | Gera a resposta final usando os chunks rerankeados como contexto |

---

## Por que cada adição importa

### Chunking Strategy (512 tokens · 50 overlap)
- Chunks muito grandes → perdem foco, o LLM dilui o contexto relevante
- Chunks muito pequenos → perdem coerência semântica
- O overlap de 50 tokens garante que frases na borda de um chunk não se percam

### Hybrid Search (BM25 + Pinecone)
- Busca semântica falha em nomes próprios, siglas e termos muito específicos
- BM25 é exato por definição — captura o que a busca vetorial perde
- RRF funde os dois rankings sem precisar calibrar pesos

### Reranker (Vertex AI Rank API)
- top-k por similaridade coseno retorna os vetores mais próximos, não necessariamente os mais úteis
- O reranker avalia cada chunk no contexto exato da query, com precisão maior
- Reduz alucinação do LLM ao filtrar chunks periféricos antes de montar o contexto
```
