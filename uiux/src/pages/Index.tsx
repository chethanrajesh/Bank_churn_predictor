import Hero from "@/components/Hero";
import Features from "@/components/Features";
import Stats from "@/components/Stats";
import InteractiveChart from "@/components/InteractiveChart";

const Index = () => {
  return (
    <div className="min-h-screen">
      <Hero />
      <Stats />
      <Features />
      <InteractiveChart />
    </div>
  );
};

export default Index;
