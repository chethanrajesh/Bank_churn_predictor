import { useEffect, useState } from "react";
import { Card } from "@/components/ui/card";

const stats = [
  { value: 10000, suffix: "+", label: "Data Points Analyzed" },
  { value: 99.9, suffix: "%", label: "Accuracy Rate" },
  { value: 500, suffix: "+", label: "Enterprise Clients" },
  { value: 24, suffix: "/7", label: "Real-time Monitoring" },
];

const Stats = () => {
  const [counts, setCounts] = useState(stats.map(() => 0));

  useEffect(() => {
    const duration = 2000;
    const steps = 60;
    const intervals = stats.map((stat, index) => {
      const increment = stat.value / steps;
      let current = 0;
      
      return setInterval(() => {
        current += increment;
        if (current >= stat.value) {
          setCounts(prev => {
            const newCounts = [...prev];
            newCounts[index] = stat.value;
            return newCounts;
          });
          clearInterval(intervals[index]);
        } else {
          setCounts(prev => {
            const newCounts = [...prev];
            newCounts[index] = current;
            return newCounts;
          });
        }
      }, duration / steps);
    });

    return () => intervals.forEach(interval => clearInterval(interval));
  }, []);

  return (
    <section className="py-24 px-4 sm:px-6 lg:px-8 relative overflow-hidden">
      <div className="absolute inset-0 bg-gradient-to-r from-primary/10 via-accent/10 to-primary/10" />
      
      <div className="container mx-auto relative z-10">
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
          {stats.map((stat, index) => (
            <Card
              key={index}
              className="p-8 text-center hover:shadow-elegant transition-all duration-300 hover:-translate-y-1 border-2 bg-card/80 backdrop-blur-sm animate-scale-in"
              style={{ animationDelay: `${index * 100}ms` }}
            >
              <div className="text-4xl md:text-5xl font-bold mb-2 bg-gradient-to-r from-primary to-accent bg-clip-text text-transparent">
                {Math.round(counts[index] * 10) / 10}{stat.suffix}
              </div>
              <div className="text-muted-foreground font-medium">{stat.label}</div>
            </Card>
          ))}
        </div>
      </div>
    </section>
  );
};

export default Stats;
