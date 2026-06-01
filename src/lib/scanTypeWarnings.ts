const KEYWORDS = {
  brainMri: ["brain", "mri", "tumor", "alz", "alzheimer", "scan", "t1", "t2"],
  chestXray: ["chest", "xray", "x-ray", "lung", "pneumonia", "cxr"],
};

const hasKeyword = (value: string, keywords: string[]) =>
  keywords.some((keyword) => value.includes(keyword));

export function getScanTypeWarning(
  fileName: string,
  expectedType: "brain-mri" | "chest-xray",
): string | null {
  const normalized = fileName.toLowerCase();

  if (expectedType === "brain-mri" && hasKeyword(normalized, KEYWORDS.chestXray)) {
    return "Uploaded image does not match expected scan type. This module expects a brain MRI, but the filename looks like a chest X-ray or lung scan.";
  }

  if (expectedType === "chest-xray" && hasKeyword(normalized, KEYWORDS.brainMri)) {
    return "Uploaded image does not match expected scan type. This module expects a chest X-ray, but the filename looks like a brain MRI scan.";
  }

  return null;
}
