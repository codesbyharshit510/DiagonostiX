import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { LucideIcon } from "lucide-react";
import { useNavigate } from "react-router-dom";

interface DiseaseCardProps {
  title: string;
  description: string;
  icon: LucideIcon;
  route: string;
  gradient: string;
}

export const DiseaseCard = ({ title, description, icon: Icon, route, gradient }: DiseaseCardProps) => {
  const navigate = useNavigate();

  return (
    <Card className="group hover:shadow-large transition-all duration-300 cursor-pointer overflow-hidden border-2 hover:border-primary/50">
      <div className={`h-2 ${gradient}`} />
      <CardHeader className="space-y-3">
        <div className="flex items-center justify-between">
          <div className={`p-3 rounded-xl ${gradient} bg-opacity-10 group-hover:scale-110 transition-transform duration-300`}>
            <Icon className="h-6 w-6 text-primary" />
          </div>
        </div>
        <CardTitle className="text-xl">{title}</CardTitle>
        <CardDescription className="text-sm">{description}</CardDescription>
      </CardHeader>
      <CardContent>
        <Button 
          onClick={() => navigate(route)} 
          className="w-full group-hover:bg-primary-dark transition-colors"
        >
          Start Detection
        </Button>
      </CardContent>
    </Card>
  );
};
