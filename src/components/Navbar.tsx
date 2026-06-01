import { Activity, LogOut, User, Menu, MessageCircle, FileText } from "lucide-react";
import { Button } from "@/components/ui/button";
import { supabase } from "@/integrations/supabase/client";
import { useNavigate, useLocation } from "react-router-dom";
import { toast } from "sonner";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { useState } from "react";
import { ChatbotModal } from "./ChatbotModal";

interface NavbarProps {
  user: any;
}

export const Navbar = ({ user }: NavbarProps) => {
  const navigate = useNavigate();
  const location = useLocation();
  const [chatbotOpen, setChatbotOpen] = useState(false);

  const handleLogout = async () => {
    const { error } = await supabase.auth.signOut();
    if (error) {
      toast.error("Error signing out");
    } else {
      toast.success("Signed out successfully");
      navigate("/auth");
    }
  };

  return (
    <>
      <nav className="sticky top-0 z-50 border-b bg-card/95 backdrop-blur supports-[backdrop-filter]:bg-card/60">
        <div className="container mx-auto px-4 h-16 flex items-center justify-between">
          <div className="flex items-center space-x-3 cursor-pointer" onClick={() => navigate("/")}>
            <div className="p-2 bg-primary rounded-lg">
              <Activity className="h-6 w-6 text-primary-foreground" />
            </div>
            <div>
              <h1 className="text-xl font-bold">DIAGNOSTIX</h1>
              <p className="text-xs text-muted-foreground">AI Medical Diagnostics</p>
            </div>
          </div>

          <div className="flex items-center space-x-3">
            <Button
              variant={location.pathname === '/results' ? "default" : "ghost"}
              size="sm"
              onClick={() => navigate('/results')}
              className="gap-2 hidden sm:flex"
            >
              <FileText className="h-4 w-4" />
              Results
            </Button>
            
            <Button
              variant="outline"
              size="sm"
              onClick={() => setChatbotOpen(true)}
              className="gap-2"
            >
              <MessageCircle className="h-4 w-4" />
              <span className="hidden sm:inline">Assistant</span>
            </Button>

            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="outline" size="icon">
                  <Menu className="h-5 w-5" />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end" className="w-56">
                <DropdownMenuLabel>My Account</DropdownMenuLabel>
                <DropdownMenuSeparator />
                <DropdownMenuItem onClick={() => navigate('/results')} className="cursor-pointer sm:hidden">
                  <FileText className="mr-2 h-4 w-4" />
                  <span>Results</span>
                </DropdownMenuItem>
                <DropdownMenuItem className="cursor-pointer">
                  <User className="mr-2 h-4 w-4" />
                  <span>{user?.email}</span>
                </DropdownMenuItem>
                <DropdownMenuSeparator />
                <DropdownMenuItem onClick={handleLogout} className="cursor-pointer text-destructive">
                  <LogOut className="mr-2 h-4 w-4" />
                  <span>Logout</span>
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </div>
        </div>
      </nav>

      <ChatbotModal open={chatbotOpen} onOpenChange={setChatbotOpen} />
    </>
  );
};
