import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { TrendingUp, Shield, BarChart3, Zap, Brain, Lock } from "lucide-react";

const features = [
  {
    icon: TrendingUp,
    title: "Real-time Analytics",
    description: "Monitor risk factors with live data updates and instant notifications",
    gradient: "from-blue-500 to-cyan-500",
  },
  {
    icon: Shield,
    title: "Risk Mitigation",
    description: "Proactive strategies to identify and minimize potential threats",
    gradient: "from-purple-500 to-pink-500",
  },
  {
    icon: BarChart3,
    title: "Advanced Reporting",
    description: "Comprehensive dashboards with customizable metrics and KPIs",
    gradient: "from-orange-500 to-red-500",
  },
  {
    icon: Brain,
    title: "AI-Powered Insights",
    description: "Machine learning algorithms to predict and prevent risks",
    gradient: "from-green-500 to-emerald-500",
  },
  {
    icon: Zap,
    title: "Lightning Fast",
    description: "Process millions of data points in seconds with optimized performance",
    gradient: "from-yellow-500 to-orange-500",
  },
  {
    icon: Lock,
    title: "Enterprise Security",
    description: "Bank-grade encryption and compliance with global standards",
    gradient: "from-indigo-500 to-purple-500",
  },
];

const Features = () => {
  return (
    <section className="py-24 px-4 sm:px-6 lg:px-8 bg-gradient-to-b from-background to-secondary/20">
      <div className="container mx-auto">
        <div className="text-center mb-16 animate-fade-in">
          <h2 className="text-4xl md:text-5xl font-bold mb-4">
            Powerful Features
          </h2>
          <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
            Everything you need to manage and analyze risks effectively
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {features.map((feature, index) => (
            <Card
              key={index}
              className="group hover:shadow-elegant transition-all duration-300 hover:-translate-y-1 border-2 hover:border-primary/50 animate-fade-in bg-card/50 backdrop-blur-sm"
              style={{ animationDelay: `${index * 100}ms` }}
            >
              <CardHeader>
                <div className={`w-12 h-12 rounded-xl bg-gradient-to-br ${feature.gradient} flex items-center justify-center mb-4 group-hover:scale-110 transition-transform`}>
                  <feature.icon className="w-6 h-6 text-white" />
                </div>
                <CardTitle className="text-xl">{feature.title}</CardTitle>
              </CardHeader>
              <CardContent>
                <CardDescription className="text-base">
                  {feature.description}
                </CardDescription>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    </section>
  );
};

export default Features;
