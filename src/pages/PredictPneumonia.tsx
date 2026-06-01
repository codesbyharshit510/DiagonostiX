import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { supabase } from "@/integrations/supabase/client";
import { Navbar } from "@/components/Navbar";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Upload, ArrowLeft, Activity } from "lucide-react";
import { toast } from "sonner";
import { API_BASE } from "@/lib/utils";
import { getScanTypeWarning } from "@/lib/scanTypeWarnings";

const PredictPneumonia = () => {
  const [user, setUser] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [scanWarning, setScanWarning] = useState<string | null>(null);
  const navigate = useNavigate();

  // ---------------- AUTH CHECK ----------------
  useEffect(() => {
    const { data: { subscription } } = supabase.auth.onAuthStateChange(
      (_event, session) => {
        setUser(session?.user ?? null);
        if (!session?.user) navigate("/auth");
      }
    );

    supabase.auth.getSession().then(({ data: { session } }) => {
      setUser(session?.user ?? null);
      if (!session?.user) navigate("/auth");
    });

    return () => subscription.unsubscribe();
  }, [navigate]);

  // ---------------- FILE SELECT ----------------
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files?.[0]) return;
    const img = e.target.files[0];

    if (!img.type.startsWith("image/")) {
      toast.error("Please upload an image");
      return;
    }

    const warning = getScanTypeWarning(img.name, "chest-xray");
    setScanWarning(warning);
    if (warning) toast.warning(warning);
    setFile(img);
  };

  // ---------------- SUBMIT ----------------
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!file) {
      toast.error("Please upload an X-ray");
      return;
    }

    setLoading(true);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const url = new URL(`${API_BASE}/image/report`);
      url.searchParams.set("disease", "pneumonia");
      url.searchParams.set("user_id", user.id);

      const response = await fetch(url.toString(), {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        toast.error("Backend error — report not generated");
        console.log(await response.text());
        return;
      }

      const html = await response.text();
      const blob = new Blob([html], { type: "text/html" });
      const blobUrl = URL.createObjectURL(blob);
      window.open(blobUrl, "_blank");

      toast.success("Pneumonia report generated!");
    } catch (err) {
      console.error(err);
      toast.error("Some error occurred");
    } finally {
      setLoading(false);
    }
  };

  // ---------------- UI ----------------
  if (!user) return null;

  return (
    <div className="min-h-screen bg-gradient-to-br from-primary/5 via-background to-secondary/5">
      <Navbar user={user} />

      <main className="container mx-auto px-4 py-12">
        <div className="max-w-3xl mx-auto space-y-6">
          <Button variant="ghost" onClick={() => navigate("/")}>
            <ArrowLeft className="mr-2 h-4 w-4" /> Back
          </Button>

          <Card className="shadow-large">
            <CardHeader>
              <div className="flex items-center gap-3 mb-4">
                <div className="p-3 bg-gradient-to-br from-teal-500 to-green-600 rounded-xl">
                  <Activity className="h-6 w-6 text-white" />
                </div>
                <div>
                  <CardTitle>Pneumonia Detection</CardTitle>
                  <CardDescription>Upload an X-ray to generate Grad-CAM</CardDescription>
                </div>
              </div>
            </CardHeader>

            <CardContent>
              <form onSubmit={handleSubmit} className="space-y-6">

                <div>
                  <Label>Chest X-ray</Label>
                  <div className="border-2 border-dashed rounded-lg p-8 text-center">
                    <Upload className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
                    <Input
                      id="xray"
                      type="file"
                      accept="image/*"
                      onChange={handleFileChange}
                      className="hidden"
                    />
                    <Label htmlFor="xray" className="cursor-pointer">
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

                <Button className="w-full" disabled={loading || !file}>
                  {loading ? "Analyzing…" : "Generate Report"}
                </Button>
              </form>
            </CardContent>
          </Card>
        </div>
      </main>
    </div>
  );
};

export default PredictPneumonia;
