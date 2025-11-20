import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import Navbar from "./components/Navbar";
import Index from "./pages/Index";
import ExecutiveOverview from "./pages/ExecutiveOverview";
import RiskAnalysis from "./pages/RiskAnalysis";
import DriverAnalysis from "./pages/DriverAnalysis";
import IndividualAnalysis from "./pages/IndividualAnalysis";
import BusinessImpact from "./pages/BusinessImpact";
import NotFound from "./pages/NotFound";

const queryClient = new QueryClient();

const App = () => (
  <QueryClientProvider client={queryClient}>
    <TooltipProvider>
      <Toaster />
      <Sonner />
      <BrowserRouter>
        <Navbar />
        <Routes>
          <Route path="/" element={<Index />} />
          <Route path="/executive-overview" element={<ExecutiveOverview />} />
          <Route path="/risk-analysis" element={<RiskAnalysis />} />
          <Route path="/driver-analysis" element={<DriverAnalysis />} />
          <Route path="/individual-analysis" element={<IndividualAnalysis />} />
          <Route path="/business-impact" element={<BusinessImpact />} />
          <Route path="*" element={<NotFound />} />
        </Routes>
      </BrowserRouter>
    </TooltipProvider>
  </QueryClientProvider>
);

export default App;
