import dotenv from "dotenv";
import fs from "fs";
import path from "path";
import OpenAI from "openai";
import { fileURLToPath } from "url";

dotenv.config();
console.log("EMBEDDING KEY:", process.env.OPENAI_API_KEY?.slice(0, 20));
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY?.trim()
});

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const policiesPath = path.join(__dirname, "knowledge-base", "policies.json");
const outputPath = path.join(__dirname, "knowledge-base", "policy-embeddings.json");

const knowledgeBase = JSON.parse(fs.readFileSync(policiesPath, "utf-8"));

function buildPolicyChunks() {
  const chunks = [];

  for (const policy of knowledgeBase.policies) {
    for (const section of policy.sections) {
      chunks.push({
        policy_id: policy.id,
        policy_name: policy.name,
        section_title: section.title,
        content: section.content,
        text: `
Policy Name: ${policy.name}
Section: ${section.title}
Content: ${section.content}
        `.trim()
      });
    }
  }

  return chunks;
}

async function createEmbedding(text) {
  const response = await openai.embeddings.create({
    model: "text-embedding-3-small",
    input: text
  });

  return response.data[0].embedding;
}

async function main() {
  console.log("Building policy embeddings...");

  const chunks = buildPolicyChunks();
  const embeddedChunks = [];

  for (let i = 0; i < chunks.length; i++) {
    const chunk = chunks[i];

    console.log(`Embedding ${i + 1}/${chunks.length}: ${chunk.policy_name} - ${chunk.section_title}`);

    const embedding = await createEmbedding(chunk.text);

    embeddedChunks.push({
      ...chunk,
      embedding
    });
  }

  fs.writeFileSync(outputPath, JSON.stringify(embeddedChunks, null, 2), "utf-8");

  console.log("Embeddings created successfully.");
  console.log(`Saved to: ${outputPath}`);
}

main().catch((error) => {
  console.error("Failed to build embeddings:", error);
});