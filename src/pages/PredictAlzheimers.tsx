import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { supabase } from "@/integrations/supabase/client";
import { Navbar } from "@/components/Navbar";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Upload, ArrowLeft, Brain } from "lucide-react";
import { toast } from "sonner";
import { API_BASE } from "@/lib/utils";
import { getScanTypeWarning } from "@/lib/scanTypeWarnings";

const PredictAlzheimers = () => {
  const [user, setUser] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [scanWarning, setScanWarning] = useState<string | null>(null);
  const navigate = useNavigate();

  // ---------------------------
  // AUTH CHECK
  // ---------------------------
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

  // ---------------------------
  // FILE SELECT
  // ---------------------------
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selected = e.target.files?.[0];
    if (!selected) return;

    if (!selected.type.startsWith("image/")) {
      toast.error("Please upload a valid MRI image file");
      return;
    }

    const warning = getScanTypeWarning(selected.name, "brain-mri");
    setScanWarning(warning);
    if (warning) toast.warning(warning);
    setFile(selected);
  };

  // ---------------------------
  // SUBMIT → GENERATE REPORT
  // ---------------------------
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!file) {
      toast.error("Please upload an MRI image");
      return;
    }

    setLoading(true);

    try {
      const formData = new FormData();
      formData.append("file", file);

      // Call your FastAPI Grad-CAM report endpoint
      const url = new URL(`${API_BASE}/image/report`);
      url.searchParams.set("disease", "alzheimers");
      url.searchParams.set("user_id", user.id);

      const response = await fetch(url.toString(), {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        toast.error("Failed to generate Alzheimer's Grad-CAM report");
        return;
      }

      // Backend returns FULL HTML
      const html = await response.text();

      // Open report in new tab
      const blob = new Blob([html], { type: "text/html" });
      const blobUrl = URL.createObjectURL(blob);
      window.open(blobUrl, "_blank");

      toast.success("Alzheimer's Grad-CAM Report Generated");
    } catch (error) {
      console.error(error);
      toast.error("Something went wrong. Try again.");
    } finally {
      setLoading(false);
    }
  };

  if (!user) return null;

  // ---------------------------
  // UI SECTION
  // ---------------------------
  return (
    <div className="min-h-screen bg-gradient-to-br from-primary/5 via-background to-secondary/5">
      <Navbar user={user} />

      <main className="container mx-auto px-4 py-12">
        <div className="max-w-3xl mx-auto space-y-6">

          <Button variant="ghost" onClick={() => navigate("/")} className="mb-4">
            <ArrowLeft className="mr-2 h-4 w-4" />
            Back to Dashboard
          </Button>

          <Card className="shadow-large">
            <CardHeader>
              <div className="flex items-center gap-3 mb-4">
                <div className="p-3 bg-gradient-to-br from-purple-500 to-purple-700 rounded-xl">
                  <Brain className="h-6 w-6 text-white" />
                </div>
                <div>
                  <CardTitle className="text-2xl">Alzheimer's Detection</CardTitle>
                  <CardDescription>
                    Upload an MRI scan to generate a Grad-CAM heatmap report
                  </CardDescription>
                </div>
              </div>
            </CardHeader>

            <CardContent>
              <form onSubmit={handleSubmit} className="space-y-6">

                {/* File Upload */}
                <div className="space-y-2">
                  <Label htmlFor="mri">Brain MRI Scan</Label>

                  <div className="border-2 border-dashed border-border rounded-lg p-8 text-center hover:border-primary transition-colors">
                    <Upload className="h-12 w-12 mx-auto text-muted-foreground mb-4" />

                    <Input
                      id="mri"
                      type="file"
                      accept="image/*"
                      onChange={handleFileChange}
                      className="hidden"
                    />

                    <Label htmlFor="mri" className="cursor-pointer">
                      <span className="text-primary hover:underline">Click to upload</span>
                      <span className="text-muted-foreground"> or drag & drop</span>
                    </Label>

                    <p className="text-sm text-muted-foreground mt-2">
                      JPG, JPEG, PNG • up to 10MB
                    </p>

                    {file && (
                      <p className="text-sm text-primary mt-2 font-medium">
                        Selected: {file.name}
                      </p>
                    )}

                    {scanWarning && (
                      <p className="mt-3 rounded-md border border-amber-300 bg-amber-50 px-3 py-2 text-sm text-amber-900">
                        {scanWarning}
                      </p>
                    )}
                  </div>
                </div>

                {/* Info Section */}
                <div className="bg-muted/50 rounded-lg p-4 space-y-2">
                  <h4 className="font-semibold text-sm">About This Detection</h4>
                  <ul className="text-sm text-muted-foreground space-y-1">
                    <li>• Detects Alzheimer's patterns from MRI scans</li>
                    <li>• Grad-CAM highlights structural brain changes</li>
                    <li>• Provides a detailed interactive HTML report</li>
                    <li>• Shows early-stage vs advanced indications</li>
                  </ul>
                </div>

                {/* Submit Button */}
                <Button
                  type="submit"
                  className="w-full"
                  disabled={loading || !file}
                >
                  {loading ? "Analyzing..." : "Generate Grad-CAM Report"}
                </Button>
              </form>

              {/* Safety Warning */}
              <div className="mt-6 p-4 bg-yellow-50 dark:bg-yellow-950/20 border border-yellow-200 dark:border-yellow-800 rounded-lg">
                <p className="text-sm text-yellow-800 dark:text-yellow-200">
                  <strong>Note:</strong> This tool is for research and educational use.
                  Always consult certified clinicians before drawing conclusions.
                </p>
              </div>

            </CardContent>
          </Card>
        </div>
      </main>
    </div>
  );
};

export default PredictAlzheimers;
