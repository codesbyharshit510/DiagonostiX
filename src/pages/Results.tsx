import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { supabase } from "@/integrations/supabase/client";
import { Navbar } from "@/components/Navbar";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, PieChart, Pie, Cell } from "recharts";
import { Download, FileText, Calendar, TrendingUp, Activity } from "lucide-react";
import { toast } from "sonner";
import { format } from "date-fns";

interface Prediction {
  id: string;
  disease_type: string;
  prediction_result: any;
  confidence_score: number;
  created_at: string;
  explainability_data?: any;
}

const COLORS = ['hsl(var(--primary))', 'hsl(var(--secondary))', 'hsl(var(--accent))', 'hsl(var(--muted))'];
const IMAGE_DISEASES = new Set(["brain_tumor", "alzheimers", "pneumonia"]);

const normalizeDiseaseType = (value: string) => {
  if (value === "heart") return "heart_disease";
  return value;
};

const formatDiseaseLabel = (value: string) =>
  normalizeDiseaseType(value).replace(/_/g, " ").toUpperCase();

const Results = () => {
  const [user, setUser] = useState<any>(null);
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [filteredPredictions, setFilteredPredictions] = useState<Prediction[]>([]);
  const [loading, setLoading] = useState(true);
  const [diseaseFilter, setDiseaseFilter] = useState<string>("all");
  const navigate = useNavigate();

  useEffect(() => {
    const { data: { subscription } } = supabase.auth.onAuthStateChange((event, session) => {
      setUser(session?.user ?? null);
      if (!session?.user) {
        navigate("/auth");
      }
    });

    supabase.auth.getSession().then(({ data: { session } }) => {
      setUser(session?.user ?? null);
      if (!session?.user) {
        navigate("/auth");
      }
    });

    return () => subscription.unsubscribe();
  }, [navigate]);

  useEffect(() => {
    if (user) {
      fetchPredictions();
    }
  }, [user]);

  useEffect(() => {
    if (diseaseFilter === "all") {
      setFilteredPredictions(predictions);
    } else {
      setFilteredPredictions(
        predictions.filter((p) => normalizeDiseaseType(p.disease_type) === diseaseFilter),
      );
    }
  }, [diseaseFilter, predictions]);

  const fetchPredictions = async () => {
    setLoading(true);
    try {
      const { data, error } = await supabase
        .from('predictions')
        .select('*')
        .eq('user_id', user.id)
        .order('created_at', { ascending: false });

      if (error) throw error;
      setPredictions(data || []);
      setFilteredPredictions(data || []);
    } catch (error) {
      toast.error("Failed to load predictions");
      console.error(error);
    } finally {
      setLoading(false);
    }
  };

  const downloadReport = (prediction: Prediction) => {
    const report = {
      disease: formatDiseaseLabel(prediction.disease_type),
      date: format(new Date(prediction.created_at), 'PPpp'),
      confidence: `${prediction.confidence_score}%`,
      results: prediction.prediction_result,
      explainability: prediction.explainability_data
    };

    const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${prediction.disease_type}_report_${prediction.id}.json`;
    a.click();
    URL.revokeObjectURL(url);
    toast.success("Report downloaded successfully");
  };

  const downloadAllReports = () => {
    const allReports = filteredPredictions.map(p => ({
      id: p.id,
      disease: normalizeDiseaseType(p.disease_type),
      date: p.created_at,
      confidence: p.confidence_score,
      results: p.prediction_result
    }));

    const blob = new Blob([JSON.stringify(allReports, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `all_predictions_${format(new Date(), 'yyyy-MM-dd')}.json`;
    a.click();
    URL.revokeObjectURL(url);
    toast.success("All reports downloaded");
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 80) return "text-green-600";
    if (confidence >= 60) return "text-yellow-600";
    return "text-red-600";
  };

  const getDiseaseStats = () => {
    const stats = predictions.reduce((acc, p) => {
      const disease = normalizeDiseaseType(p.disease_type);
      acc[disease] = (acc[disease] || 0) + 1;
      return acc;
    }, {} as Record<string, number>);

    return Object.entries(stats).map(([name, value]) => ({
      name: formatDiseaseLabel(name),
      value
    }));
  };

  const getConfidenceTrend = () => {
    return filteredPredictions.slice(0, 10).reverse().map((p, idx) => ({
      name: `Test ${idx + 1}`,
      confidence: p.confidence_score,
      date: format(new Date(p.created_at), 'MMM dd')
    }));
  };

  const getAvgConfidence = () => {
    if (filteredPredictions.length === 0) return 0;
    const sum = filteredPredictions.reduce((acc, p) => acc + (p.confidence_score || 0), 0);
    return (sum / filteredPredictions.length).toFixed(1);
  };

  if (!user || loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-primary/5 via-background to-secondary/5">
        <Navbar user={user} />
        <div className="container mx-auto px-4 py-12 flex items-center justify-center">
          <div className="text-center">Loading results...</div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-primary/5 via-background to-secondary/5">
      <Navbar user={user} />
      
      <main className="container mx-auto px-4 py-12">
        <div className="max-w-7xl mx-auto space-y-6">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold">Prediction Results</h1>
              <p className="text-muted-foreground mt-1">
                View and analyze your disease prediction history
              </p>
            </div>
            <Button onClick={downloadAllReports} disabled={filteredPredictions.length === 0}>
              <Download className="mr-2 h-4 w-4" />
              Download All
            </Button>
          </div>

          {/* Stats Overview */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="text-sm font-medium">Total Tests</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{predictions.length}</div>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="text-sm font-medium">Avg Confidence</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{getAvgConfidence()}%</div>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="text-sm font-medium">This Month</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">
                  {predictions.filter(p => 
                    new Date(p.created_at).getMonth() === new Date().getMonth()
                  ).length}
                </div>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="text-sm font-medium">Disease Types</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{getDiseaseStats().length}</div>
              </CardContent>
            </Card>
          </div>

          {/* Filter */}
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle>Filter Results</CardTitle>
                <Select value={diseaseFilter} onValueChange={setDiseaseFilter}>
                  <SelectTrigger className="w-[200px]">
                    <SelectValue placeholder="Select disease" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">All Diseases</SelectItem>
                    <SelectItem value="brain_tumor">Brain Tumor</SelectItem>
                    <SelectItem value="alzheimers">Alzheimer's</SelectItem>
                    <SelectItem value="pneumonia">Pneumonia</SelectItem>
                    <SelectItem value="diabetes">Diabetes</SelectItem>
                    <SelectItem value="heart_disease">Heart Disease</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </CardHeader>
          </Card>

          <Tabs defaultValue="overview" className="space-y-4">
            <TabsList>
              <TabsTrigger value="overview">Overview</TabsTrigger>
              <TabsTrigger value="history">History</TabsTrigger>
              <TabsTrigger value="analytics">Analytics</TabsTrigger>
            </TabsList>

            <TabsContent value="overview" className="space-y-4">
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <TrendingUp className="h-5 w-5" />
                      Confidence Trend
                    </CardTitle>
                    <CardDescription>Recent prediction confidence scores</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <ResponsiveContainer width="100%" height={300}>
                      <LineChart data={getConfidenceTrend()}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="name" />
                        <YAxis domain={[0, 100]} />
                        <Tooltip />
                        <Legend />
                        <Line 
                          type="monotone" 
                          dataKey="confidence" 
                          stroke="hsl(var(--primary))" 
                          strokeWidth={2}
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Activity className="h-5 w-5" />
                      Tests by Disease
                    </CardTitle>
                    <CardDescription>Distribution of predictions</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <ResponsiveContainer width="100%" height={300}>
                      <PieChart>
                        <Pie
                          data={getDiseaseStats()}
                          cx="50%"
                          cy="50%"
                          labelLine={false}
                          label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                          outerRadius={80}
                          fill="#8884d8"
                          dataKey="value"
                        >
                          {getDiseaseStats().map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                          ))}
                        </Pie>
                        <Tooltip />
                      </PieChart>
                    </ResponsiveContainer>
                  </CardContent>
                </Card>
              </div>
            </TabsContent>

            <TabsContent value="history" className="space-y-4">
              {filteredPredictions.length === 0 ? (
                <Card>
                  <CardContent className="py-12 text-center">
                    <FileText className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
                    <p className="text-muted-foreground">No predictions found</p>
                  </CardContent>
                </Card>
              ) : (
                filteredPredictions.map((prediction) => (
                  <Card key={prediction.id}>
                    <CardHeader>
                      <div className="flex items-start justify-between">
                        <div>
                          <CardTitle className="flex items-center gap-2">
                            {formatDiseaseLabel(prediction.disease_type)}
                            <Badge variant="outline">
                              {IMAGE_DISEASES.has(normalizeDiseaseType(prediction.disease_type)) ? 'Image' : 'Tabular'}
                            </Badge>
                          </CardTitle>
                          <CardDescription className="flex items-center gap-2 mt-1">
                            <Calendar className="h-3 w-3" />
                            {format(new Date(prediction.created_at), 'PPpp')}
                          </CardDescription>
                        </div>
                        <Button variant="outline" size="sm" onClick={() => downloadReport(prediction)}>
                          <Download className="h-4 w-4 mr-2" />
                          Download
                        </Button>
                      </div>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-4">
                        <div>
                          <p className="text-sm font-medium mb-2">Confidence Score</p>
                          <div className="flex items-center gap-2">
                            <div className="flex-1 bg-muted rounded-full h-2">
                              <div 
                                className="bg-primary h-2 rounded-full transition-all"
                                style={{ width: `${prediction.confidence_score}%` }}
                              />
                            </div>
                            <span className={`text-lg font-bold ${getConfidenceColor(prediction.confidence_score)}`}>
                              {prediction.confidence_score}%
                            </span>
                          </div>
                        </div>

                        <div>
                          <p className="text-sm font-medium mb-2">Prediction Results</p>
                          <div className="bg-muted/50 rounded-lg p-4">
                            <pre className="text-xs overflow-auto max-h-48">
                              {JSON.stringify(prediction.prediction_result, null, 2)}
                            </pre>
                          </div>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                ))
              )}
            </TabsContent>

            <TabsContent value="analytics" className="space-y-4">
              <Card>
                <CardHeader>
                  <CardTitle>Confidence Distribution</CardTitle>
                  <CardDescription>Analysis of prediction confidence levels</CardDescription>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={400}>
                    <BarChart data={getConfidenceTrend()}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="name" />
                      <YAxis domain={[0, 100]} />
                      <Tooltip />
                      <Legend />
                      <Bar dataKey="confidence" fill="hsl(var(--primary))" />
                    </BarChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        </div>
      </main>
    </div>
  );
};

export default Results;
