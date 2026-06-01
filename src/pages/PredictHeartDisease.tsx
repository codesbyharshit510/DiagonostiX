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
import { ArrowLeft, Heart } from "lucide-react";
import { toast } from "sonner";
import { API_BASE } from "@/lib/utils";

const PredictHeartDisease = () => {
  const [user, setUser] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  // Heart Disease model input fields
  const [form, setForm] = useState({
    age: "",
    sex: "",
    cp: "",
    trestbps: "",
    chol: "",
    fbs: "",
    restecg: "",
    thalach: "",
    exang: "",
    oldpeak: "",
    slope: "",
    ca: "",
    thal: "",
  });

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

  // Update form values
  const updateField = (key: string, value: string) => {
    setForm((prev) => ({ ...prev, [key]: value }));
  };

  // ---------------------------
  // SUBMIT → Generate SHAP Report
  // ---------------------------
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    const missing = Object.entries(form).some(([_, v]) => v.trim() === "");
    if (missing) {
      toast.error("Please fill all fields");
      return;
    }

    setLoading(true);

    try {
      const payload = Object.fromEntries(
        Object.entries(form).map(([k, v]) => [k, parseFloat(v)])
      );
      const requestUrl = new URL(`${API_BASE}/tabular/explain`);
      requestUrl.searchParams.set("disease", "heart");
      requestUrl.searchParams.set("user_id", user.id);

      const response = await fetch(
        requestUrl.toString(),
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        }
      );

      if (!response.ok) {
        toast.error("Failed to generate SHAP report");
        return;
      }

      const html = await response.text();

      // Open HTML report in new tab
      const blob = new Blob([html], { type: "text/html" });
      const url = URL.createObjectURL(blob);
      window.open(url, "_blank");

      toast.success("Heart Disease SHAP Report Generated");
    } catch (error) {
      console.error(error);
      toast.error("Something went wrong");
    } finally {
      setLoading(false);
    }
  };

  if (!user) return null;

  // ---------------------------
  // UI
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
                <div className="p-3 bg-gradient-to-br from-red-500 to-pink-600 rounded-xl">
                  <Heart className="h-6 w-6 text-white" />
                </div>
                <div>
                  <CardTitle className="text-2xl">Heart Disease Prediction</CardTitle>
                  <CardDescription>
                    Enter cardiovascular metrics to generate a SHAP explanation report
                  </CardDescription>
                </div>
              </div>
            </CardHeader>

            <CardContent>
              <form onSubmit={handleSubmit} className="space-y-6">

                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                  {Object.entries(form).map(([key, value]) => (
                    <div key={key} className="space-y-1">
                      <Label>{key.toUpperCase()}</Label>
                      <Input
                        type="number"
                        value={value}
                        onChange={(e) => updateField(key, e.target.value)}
                        placeholder={`Enter ${key}`}
                      />
                    </div>
                  ))}
                </div>

                <Button className="w-full" type="submit" disabled={loading}>
                  {loading ? "Analyzing..." : "Generate SHAP Report"}
                </Button>
              </form>

              <div className="mt-6 p-4 bg-yellow-50 dark:bg-yellow-950/20 border border-yellow-200 dark:border-yellow-800 rounded-lg">
                <p className="text-sm text-yellow-800 dark:text-yellow-200">
                  <strong>Note:</strong> SHAP explains how each input contributed to the
                  prediction. Always verify with a medical professional.
                </p>
              </div>

            </CardContent>
          </Card>
        </div>
      </main>
    </div>
  );
};

export default PredictHeartDisease;
