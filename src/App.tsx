import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import Index from "./pages/Index";
import Auth from "./pages/Auth";
import PredictBrainTumor from "./pages/PredictBrainTumor";
import PredictAlzheimers from "./pages/PredictAlzheimers";
import PredictPneumonia from "./pages/PredictPneumonia";
import PredictDiabetes from "./pages/PredictDiabetes";
import PredictHeartDisease from "./pages/PredictHeartDisease";
import Results from "./pages/Results";
import NotFound from "./pages/NotFound";

const queryClient = new QueryClient();

const App = () => (
  <QueryClientProvider client={queryClient}>
    <TooltipProvider>
      <Toaster />
      <Sonner />
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Index />} />
          <Route path="/auth" element={<Auth />} />
          <Route path="/predict/brain-tumor" element={<PredictBrainTumor />} />
          <Route path="/predict/alzheimers" element={<PredictAlzheimers />} />
          <Route path="/predict/pneumonia" element={<PredictPneumonia />} />
          <Route path="/predict/diabetes" element={<PredictDiabetes />} />
          <Route path="/predict/heart-disease" element={<PredictHeartDisease />} />
          <Route path="/results" element={<Results />} />
          {/* ADD ALL CUSTOM ROUTES ABOVE THE CATCH-ALL "*" ROUTE */}
          <Route path="*" element={<NotFound />} />
        </Routes>
      </BrowserRouter>
    </TooltipProvider>
  </QueryClientProvider>
);

export default App;
