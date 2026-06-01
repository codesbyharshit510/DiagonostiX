import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { supabase } from "@/integrations/supabase/client";
import { Navbar } from "@/components/Navbar";
import { DiseaseCard } from "@/components/DiseaseCard";
import { Brain, Heart, Droplets, Activity, Stethoscope } from "lucide-react";
import { Button } from "@/components/ui/button";

const Index = () => {
  const [user, setUser] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const navigate = useNavigate();

  useEffect(() => {
    const { data: { subscription } } = supabase.auth.onAuthStateChange((event, session) => {
      setUser(session?.user ?? null);
      setLoading(false);
    });

    supabase.auth.getSession().then(({ data: { session } }) => {
      setUser(session?.user ?? null);
      setLoading(false);
    });

    return () => subscription.unsubscribe();
  }, []);

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary"></div>
      </div>
    );
  }

  if (!user) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-primary/10 via-background to-secondary/10">
        <div className="container mx-auto px-4 py-16">
          <div className="max-w-4xl mx-auto text-center space-y-8">
            <div className="flex justify-center mb-6">
              <div className="p-4 bg-primary rounded-2xl shadow-large">
                <Activity className="h-16 w-16 text-primary-foreground" />
              </div>
            </div>
            
            <h1 className="text-5xl md:text-6xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-primary to-secondary">
              DIAGNOSTIX
            </h1>
            
            <p className="text-xl md:text-2xl text-muted-foreground max-w-2xl mx-auto">
              AI-Driven Multi-Disease Detection Platform
            </p>
            
            <p className="text-lg text-foreground/80 max-w-2xl mx-auto">
              Harness the power of artificial intelligence for early disease detection. 
              Our advanced ML models analyze medical imaging and health data to provide 
              accurate diagnostic insights with explainable AI.
            </p>

            <div className="flex flex-col sm:flex-row gap-4 justify-center pt-6">
              <Button 
                size="lg" 
                onClick={() => navigate("/auth")}
                className="text-lg px-8"
              >
                Get Started
              </Button>
              <Button 
                size="lg" 
                variant="outline"
                onClick={() => navigate("/auth")}
                className="text-lg px-8"
              >
                Sign In
              </Button>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 pt-12 max-w-3xl mx-auto">
              <div className="p-6 bg-card rounded-xl shadow-medium">
                <h3 className="font-semibold text-lg mb-2">5 Disease Models</h3>
                <p className="text-sm text-muted-foreground">
                  Brain Tumor, Alzheimer's, Pneumonia, Diabetes, and Heart Disease detection
                </p>
              </div>
              <div className="p-6 bg-card rounded-xl shadow-medium">
                <h3 className="font-semibold text-lg mb-2">Explainable AI</h3>
                <p className="text-sm text-muted-foreground">
                  Grad-CAM and SHAP visualizations for transparent predictions
                </p>
              </div>
              <div className="p-6 bg-card rounded-xl shadow-medium">
                <h3 className="font-semibold text-lg mb-2">Secure & Private</h3>
                <p className="text-sm text-muted-foreground">
                  Your health data is encrypted and protected with enterprise-grade security
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-primary/5 via-background to-secondary/5">
      <Navbar user={user} />
      
      <main className="container mx-auto px-4 py-12">
        <div className="max-w-6xl mx-auto space-y-8">
          <div className="text-center space-y-4">
            <h1 className="text-4xl md:text-5xl font-bold">
              Disease Detection Dashboard
            </h1>
            <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
              Select a disease detection module to begin analysis. Our AI models provide accurate predictions with explainable insights.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            <DiseaseCard
              title="Brain Tumor Detection"
              description="MRI-based detection using CNN with Grad-CAM visualization for tumor localization"
              icon={Brain}
              route="/predict/brain-tumor"
              gradient="bg-gradient-to-br from-primary to-primary-dark"
            />
            
            <DiseaseCard
              title="Alzheimer's Detection"
              description="Early-stage Alzheimer's detection from brain MRI scans using deep learning"
              icon={Brain}
              route="/predict/alzheimers"
              gradient="bg-gradient-to-br from-purple-500 to-purple-700"
            />
            
            <DiseaseCard
              title="Pneumonia Detection"
              description="Chest X-ray analysis with Grad-CAM highlighting affected lung regions"
              icon={Activity}
              route="/predict/pneumonia"
              gradient="bg-gradient-to-br from-secondary to-green-600"
            />
            
            <DiseaseCard
              title="Diabetes Prediction"
              description="Risk assessment using health metrics with SHAP explainability"
              icon={Droplets}
              route="/predict/diabetes"
              gradient="bg-gradient-to-br from-orange-500 to-red-600"
            />
            
            <DiseaseCard
              title="Heart Disease Prediction"
              description="Cardiovascular risk analysis with interpretable feature importance"
              icon={Heart}
              route="/predict/heart-disease"
              gradient="bg-gradient-to-br from-red-500 to-pink-600"
            />
            
            <div className="flex items-center justify-center p-8 border-2 border-dashed border-muted-foreground/30 rounded-xl">
              <div className="text-center space-y-2">
                <Stethoscope className="h-12 w-12 mx-auto text-muted-foreground/50" />
                <p className="text-sm text-muted-foreground">
                  More disease models may be added in the future.Stay tuned!
                </p>
              </div>
            </div>
          </div>

          <div className="bg-card rounded-xl shadow-medium p-6 mt-8">
            <h2 className="text-2xl font-semibold mb-4">How It Works</h2>
            <div className="grid md:grid-cols-3 gap-6">
              <div className="space-y-2">
                <div className="w-10 h-10 bg-primary rounded-full flex items-center justify-center text-primary-foreground font-bold">1</div>
                <h3 className="font-semibold">Upload Data</h3>
                <p className="text-sm text-muted-foreground">Upload medical images (MRI/X-Ray) or health data (CSV)</p>
              </div>
              <div className="space-y-2">
                <div className="w-10 h-10 bg-primary rounded-full flex items-center justify-center text-primary-foreground font-bold">2</div>
                <h3 className="font-semibold">AI Analysis</h3>
                <p className="text-sm text-muted-foreground">Our trained ML models process and analyze your data</p>
              </div>
              <div className="space-y-2">
                <div className="w-10 h-10 bg-primary rounded-full flex items-center justify-center text-primary-foreground font-bold">3</div>
                <h3 className="font-semibold">Get Results</h3>
                <p className="text-sm text-muted-foreground">View predictions with explainable AI visualizations</p>
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
};

export default Index;
