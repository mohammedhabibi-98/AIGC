import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import fs from "fs";
import path from "path";
import OpenAI from "openai";
import { fileURLToPath } from "url";

dotenv.config();
console.log("OPENAI KEY START:", process.env.OPENAI_API_KEY?.slice(0, 12));
console.log("OPENAI KEY LENGTH:", process.env.OPENAI_API_KEY?.length);

if (!process.env.OPENAI_API_KEY) {
  throw new Error("OPENAI_API_KEY is missing from .env");
}
const app = express();
let conversationHistory = [];
const PORT = process.env.PORT || 5000;

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY
});

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

app.use(cors());
app.use(express.json());

const knowledgeBasePath = path.join(__dirname, "policies.json");
const knowledgeBase = JSON.parse(fs.readFileSync(knowledgeBasePath, "utf-8"));

function detectLanguage(text) {
  return /[\u0600-\u06FF]/.test(text) ? "ar" : "en";
}

function normalizeText(text) {
  return text.toLowerCase().trim();
}

function checkSpecialRules(question, language) {
  const normalizedQuestion = normalizeText(question);

  if (!knowledgeBase.special_rules) return null;

  for (const rule of knowledgeBase.special_rules) {
    const matched = rule.keywords.some((keyword) =>
      normalizedQuestion.includes(normalizeText(keyword))
    );

    if (matched) {
      return language === "ar" ? rule.response_ar : rule.response_en;
    }
  }

  return null;
}



function buildContext(searchResults) {
  if (!searchResults.length) return "No relevant policy content found.";

  return searchResults
    .map((item, index) => {
      return `
Source ${index + 1}
Policy ID: ${item.policy_id}
Policy Name: ${item.policy_name}
Section: ${item.section_title}
Content: ${item.content}
`;
    })
    .join("\n");
}
function cosineSimilarity(vecA, vecB) {
  let dotProduct = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < vecA.length; i++) {
    dotProduct += vecA[i] * vecB[i];
    normA += vecA[i] * vecA[i];
    normB += vecB[i] * vecB[i];
  }

  return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}

const embeddingsPath = path.join(
  __dirname,
  "policy-embeddings.json"
);

const embeddedPolicies = JSON.parse(
  fs.readFileSync(embeddingsPath, "utf-8")
);

async function semanticSearch(question, topK = 8) {
  const response = await openai.embeddings.create({
    model: "text-embedding-3-small",
    input: question
  });

  const queryEmbedding = response.data[0].embedding;

  const scoredResults = embeddedPolicies.map((item) => ({
    ...item,
    score: cosineSimilarity(queryEmbedding, item.embedding)
  }));

  return scoredResults
    .sort((a, b) => b.score - a.score)
    .slice(0, topK);
}


app.get("/", (req, res) => {
  res.json({
    status: "running",
    service: "AIGC Policy AI Assistant Backend"
  });
});

app.post("/api/chat", async (req, res) => {
  try {
    const { question } = req.body;

    if (!question || typeof question !== "string") {
      return res.status(400).json({
        error: "Question is required."
      });
    }

    const language = detectLanguage(question);
    const normalizedQuestion = normalizeText(question);

    const policyOverviewTriggers = [
      "what are the policies",
      "what policies",
      "available policies",
      "company policies",
      "list policies",
      "ما هي السياسات",
      "ما السياسات",
      "السياسات الخاصة بالشركة",
      "السياسات المتاحة",
      "اذكر السياسات"
    ];

    if (
      policyOverviewTriggers.some((trigger) =>
        normalizedQuestion.includes(normalizeText(trigger))
      )
    ) {
      const answer =
        language === "ar"
          ? `السياسات المتاحة حالياً داخل AIGC Policy AI Assistant هي:

• سياسة القروض
• سياسة الحضور والإجازات
• سياسة سفر العمل
• سياسة تقييم الأداء
• سياسة زيادة المؤهلات

يمكنك سؤالي مثلًا عن الإجازات السنوية، الإجازات المرضية، بدل السفر، تقييم الأداء، القروض، أو زيادة المؤهلات.

ملاحظة: سياسة العمل من المنزل موقوفة حالياً.`
          : `The currently available policies in the AIGC Policy AI Assistant are:

• Loan Policy
• Attendance and Leave Policy
• Business Travel Policy
• Performance Appraisal Policy
• Qualification Increment Policy

You may ask about annual leave, sick leave, business travel, per diem, performance appraisal, loans, or qualification increments.

Note: The work-from-home policy is currently suspended.`;

      return res.json({
        answer,
        source: "Policy Overview",
        language
      });
    }

    const specialRuleAnswer = checkSpecialRules(question, language);

    if (specialRuleAnswer) {
      return res.json({
        answer: specialRuleAnswer,
        source: "Administrative Rule",
        language
      });
    }

    const lastUserQuestion =
      conversationHistory
        .filter((msg) => msg.role === "user")
        .slice(-1)[0]?.content || "";

    const retrievalQuestion = `
Previous Question:
${lastUserQuestion}

Current Question:
${question}
`;
const contextualQuestion = `
Previous Topic:
${lastUserQuestion}

User Follow-up:
${question}
`;
    const searchResults = await semanticSearch(retrievalQuestion);
    const context = buildContext(searchResults);

    const systemPrompt =
      language === "ar"
        ? `
أنت مساعد سياسات داخلي احترافي خاص بشركة AIGC.

مهمتك:
تقديم إجابات دقيقة وواضحة ومهنية اعتمادًا فقط على سياسات الشركة المعطاة لك.

القواعد:
1. إذا كان السؤال عامًا، قدّم إجابة عامة مفيدة من السياسات المتاحة.
2. إذا كان السؤال غير واضح، اطلب توضيحًا قصيرًا مع أمثلة.
إذا لم تجد إجابة مباشرة حرفية،
حاول استنتاج الإجابة اعتمادًا على السياسات ذات العلاقة.

يمكنك الربط بين أكثر من سياسة للوصول إلى إجابة مفيدة،
طالما أنك لا تخترع معلومات غير موجودة.4. أجب بنفس لغة سؤال الموظف.
5. اجعل الإجابة احترافية وواضحة وسهلة الفهم.
6. استخدم نقاطًا عند الحاجة.
7. اذكر اسم السياسة ذات العلاقة عند توفرها.
8. لا تخترع أرقامًا أو شروطًا أو إجراءات غير موجودة.
9. إذا كانت السياسة تختلف حسب الدولة، وضّح الفرق.
`
        : `
You are a professional internal policy assistant for AIGC.

Your role:
Provide accurate, professional, and policy-grounded answers based on the provided company policy context.

Rules:
1. If the question is general, provide a useful general answer from the available policies.
2. If the question is unclear, ask a short clarifying question with examples.
3. If the information is truly unavailable, say: "This information is not available in the currently provided policies."
4. Respond in the same language as the employee's question.
5. Keep answers professional, clear, and easy to understand.
6. Use bullet points when helpful.
7. Mention the relevant policy name when available.
8. Do not invent numbers, conditions, or procedures.
9. If country-specific rules exist, explain the differences clearly.
`;

    const userPrompt = `
Employee Question:
${question}

Relevant Policy Context:
${context}
`;

    conversationHistory.push({
      role: "user",
      content: question
    });

    const messages = [
      {
        role: "system",
        content: systemPrompt
      },
      ...conversationHistory.slice(-6),
      {
        role: "user",
        content: userPrompt
      }
    ];

    const completion = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      temperature: 0.3,
      messages
    });

    const answer = completion.choices[0].message.content;

    conversationHistory.push({
      role: "assistant",
      content: answer
    });

    if (conversationHistory.length > 20) {
      conversationHistory = conversationHistory.slice(-20);
    }

    res.json({
      answer,
      sources: searchResults.map((item) => ({
        policy_id: item.policy_id,
        policy_name: item.policy_name,
        section_title: item.section_title
      })),
      language
    });
  } catch (error) {
    console.error("Error:", error);

    res.status(500).json({
  error: error.message,
  code: error.code || null,
  status: error.status || null
});
  }
});

app.listen(PORT, () => {
  console.log(`AIGC Policy AI Assistant backend is running on port ${PORT}`);
});