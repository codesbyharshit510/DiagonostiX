// src/pages/PredictBrainTumor.tsx
import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { supabase } from "@/integrations/supabase/client";
import { Navbar } from "@/components/Navbar";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Upload, ArrowLeft, Brain } from "lucide-react";
import { toast } from "sonner";
import { getScanTypeWarning } from "@/lib/scanTypeWarnings";

const API_BASE = import.meta.env.VITE_API_BASE;

const PredictBrainTumor = () => {
  const [user, setUser] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [scanWarning, setScanWarning] = useState<string | null>(null);
  const navigate = useNavigate();

  useEffect(() => {
    const { data: { subscription } } = supabase.auth.onAuthStateChange((_e, s) => {
      setUser(s?.user ?? null);
      if (!s?.user) navigate("/auth");
    });

    supabase.auth.getSession().then(({ data: { session } }) => {
      setUser(session?.user ?? null);
      if (!session?.user) navigate("/auth");
    });

    return () => subscription.unsubscribe();
  }, [navigate]);

  const handleFileChange = (e: any) => {
    const f = e.target.files?.[0];
    if (!f?.type.startsWith("image/")) return toast.error("Upload an image file");
    const warning = getScanTypeWarning(f.name, "brain-mri");
    setScanWarning(warning);
    if (warning) toast.warning(warning);
    setFile(f);
  };

  const handleSubmit = async (e: any) => {
    e.preventDefault();
    if (!file) return toast.error("Upload a file");

    setLoading(true);
    try {
      const fd = new FormData();
      fd.append("file", file);

      const url = new URL(`${API_BASE}/image/report`);
      url.searchParams.set("disease", "brain_tumor");
      url.searchParams.set("user_id", user.id);

      const res = await fetch(url.toString(), {
        method: "POST",
        body: fd,
      });

      if (!res.ok) return toast.error("Failed to generate report");

      const html = await res.text();
      const blobUrl = URL.createObjectURL(new Blob([html], { type: "text/html" }));
      window.open(blobUrl, "_blank");

      toast.success("Brain Tumor Grad-CAM Report Ready!");
    } catch {
      toast.error("Something went wrong");
    }
    setLoading(false);
  };

  if (!user) return null;

  return (
    <div className="min-h-screen bg-gradient-to-br from-primary/5 via-background to-secondary/5">
      <Navbar user={user} />
      <main className="container mx-auto px-4 py-12">
        <Button variant="ghost" onClick={() => navigate("/")}>
          <ArrowLeft className="mr-2 h-4 w-4" /> Back
        </Button>

        <Card className="max-w-3xl mx-auto shadow-large mt-6">
          <CardHeader>
            <div className="flex items-center gap-3">
              <div className="p-3 bg-gradient-to-br from-purple-500 to-indigo-600 rounded-xl">
                <Brain className="h-6 w-6 text-white" />
              </div>
              <div>
                <CardTitle>Brain Tumor Detection</CardTitle>
                <CardDescription>Upload MRI scan → Grad-CAM analysis</CardDescription>
              </div>
            </div>
          </CardHeader>

          <CardContent>
            <form onSubmit={handleSubmit} className="space-y-6">
              <div>
                <Label>MRI Scan Image</Label>
                <div className="border-2 border-dashed border-border p-8 rounded-lg text-center">
                  <Upload className="mx-auto h-12 w-12 text-muted-foreground mb-4" />
                  <Input id="mri" type="file" className="hidden" onChange={handleFileChange} />
                  <Label htmlFor="mri" className="cursor-pointer">
                    <span className="text-primary">Click to upload</span>
                  </Label>

                  {file && <p className="mt-2 text-primary">{file.name}</p>}
                  {scanWarning && (
                    <p className="mt-3 rounded-md border border-amber-300 bg-amber-50 px-3 py-2 text-sm text-amber-900">
                      {scanWarning}
                    </p>
                  )}
                </div>
              </div>

              <Button disabled={loading || !file} className="w-full">
                {loading ? "Analyzing..." : "Generate Grad-CAM Report"}
              </Button>
            </form>
          </CardContent>
        </Card>
      </main>
    </div>
  );
};

export default PredictBrainTumor;
