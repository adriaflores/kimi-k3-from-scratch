import manifest from "./docs-files.json";
/** Integration-owned frozen inventory; the JSON also powers standalone QA. */
export const SOURCE_COMMIT = manifest.sourceCommit;
export const REPOSITORY = manifest.repository;
export const DOC_FILES = manifest.files.map(({ sourcePath, description }) => [sourcePath, description] as const);
export type SourceFile = (typeof manifest.files)[number]["sourcePath"];
export const docId = (sourcePath: string) => sourcePath.replace(/\.py$/, "");
export const docHref = (sourcePath: string) => `/docs/${docId(sourcePath)}/`;
export const sourceHref = (sourcePath: string) =>
  `https://github.com/${REPOSITORY}/blob/${SOURCE_COMMIT}/${sourcePath}`;
