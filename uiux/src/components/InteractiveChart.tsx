import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { TrendingUp, TrendingDown } from "lucide-react";
import { useState } from "react";

const riskData = [
  { category: "Market Risk", level: 65, trend: "up", change: "+5.2%" },
  { category: "Credit Risk", level: 42, trend: "down", change: "-2.1%" },
  { category: "Operational Risk", level: 78, trend: "up", change: "+8.5%" },
  { category: "Liquidity Risk", level: 35, trend: "down", change: "-4.3%" },
  { category: "Compliance Risk", level: 52, trend: "up", change: "+1.8%" },
];

const InteractiveChart = () => {
  const [selectedRisk, setSelectedRisk] = useState<string | null>(null);

  return (
    <section className="py-24 px-4 sm:px-6 lg:px-8 bg-gradient-to-b from-secondary/20 to-background">
      <div className="container mx-auto">
        <div className="text-center mb-16 animate-fade-in">
          <h2 className="text-4xl md:text-5xl font-bold mb-4">
            Risk Distribution Overview
          </h2>
          <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
            Interactive analysis of key risk factors across your portfolio
          </p>
        </div>

        <Card className="max-w-4xl mx-auto shadow-elegant border-2">
          <CardHeader>
            <CardTitle>Risk Factor Analysis</CardTitle>
            <CardDescription>Click on any risk category to view details</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {riskData.map((risk, index) => (
              <div
                key={index}
                className={`group cursor-pointer p-4 rounded-lg border-2 transition-all duration-300 hover:shadow-md ${
                  selectedRisk === risk.category
                    ? "border-primary bg-primary/5 shadow-md"
                    : "border-border hover:border-primary/50"
                }`}
                onClick={() => setSelectedRisk(selectedRisk === risk.category ? null : risk.category)}
              >
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-3">
                    <span className="font-semibold text-lg">{risk.category}</span>
                    <Badge variant={risk.level > 60 ? "destructive" : "secondary"}>
                      {risk.level > 60 ? "High" : risk.level > 40 ? "Medium" : "Low"}
                    </Badge>
                  </div>
                  <div className="flex items-center gap-2">
                    {risk.trend === "up" ? (
                      <TrendingUp className="w-5 h-5 text-destructive" />
                    ) : (
                      <TrendingDown className="w-5 h-5 text-green-500" />
                    )}
                    <span className={risk.trend === "up" ? "text-destructive" : "text-green-500"}>
                      {risk.change}
                    </span>
                  </div>
                </div>

                <div className="relative h-3 bg-secondary rounded-full overflow-hidden">
                  <div
                    className={`absolute inset-y-0 left-0 rounded-full transition-all duration-1000 ${
                      risk.level > 60
                        ? "bg-gradient-to-r from-destructive to-red-600"
                        : risk.level > 40
                        ? "bg-gradient-to-r from-yellow-500 to-orange-500"
                        : "bg-gradient-to-r from-green-500 to-emerald-500"
                    }`}
                    style={{ width: `${risk.level}%`, animationDelay: `${index * 100}ms` }}
                  />
                </div>

                {selectedRisk === risk.category && (
                  <div className="mt-4 p-4 bg-secondary/50 rounded-lg animate-fade-in">
                    <p className="text-sm text-muted-foreground">
                      Current risk level: <strong>{risk.level}%</strong>
                    </p>
                    <p className="text-sm text-muted-foreground mt-1">
                      This risk category has shown a <strong>{risk.change}</strong> change in the last period.
                    </p>
                  </div>
                )}
              </div>
            ))}
          </CardContent>
        </Card>
      </div>
    </section>
  );
};

export default InteractiveChart;
