const API_BASE = import.meta.env.VITE_API_BASE;

/* ---------------- IMAGE REPORT ---------------- */

export async function uploadImageReport(
  disease: string,
  file: File
) {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch(
    `${API_BASE}/image/report?disease=${disease}`,
    {
      method: "POST",
      body: formData,
    }
  );

  if (!res.ok) {
    throw new Error("Image report generation failed");
  }

  return res.text(); // HTML report
}

/* ---------------- TABULAR PREDICTION ---------------- */

export async function predictTabular(
  disease: string,
  sample: Record<string, number>,
  userId: string
) {
  const res = await fetch(
    `${API_BASE}/tabular/predict?disease=${disease}`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        sample,
        user_id: userId,
      }),
    }
  );

  if (!res.ok) {
    throw new Error("Tabular prediction failed");
  }

  return res.json();
}
