import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Slider } from "@/components/ui/slider";
import { Button } from "@/components/ui/button";
import { TrendingUp, Zap, Target, Activity, Download, CheckCircle2 } from "lucide-react";
import { useState } from "react";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar } from "recharts";

const drivers = [
  {
    name: "Economic Indicators",
    impact: 85,
    trend: "increasing",
    factors: ["GDP Growth", "Inflation Rate", "Employment Data"],
    icon: TrendingUp,
  },
  {
    name: "Market Sentiment",
    impact: 72,
    trend: "stable",
    factors: ["Consumer Confidence", "Investor Sentiment", "Media Coverage"],
    icon: Activity,
  },
  {
    name: "Regulatory Environment",
    impact: 68,
    trend: "increasing",
    factors: ["Policy Changes", "Compliance Requirements", "Legal Framework"],
    icon: Target,
  },
  {
    name: "Technological Disruption",
    impact: 91,
    trend: "increasing",
    factors: ["Innovation Rate", "Digital Transformation", "Automation"],
    icon: Zap,
  },
];

const DriverAnalysis = () => {
  const [selectedDriver, setSelectedDriver] = useState<string | null>(null);
  const [impactWeights, setImpactWeights] = useState<{ [key: string]: number }>({
    "Economic Indicators": 85,
    "Market Sentiment": 72,
    "Regulatory Environment": 68,
    "Technological Disruption": 91,
  });

  const featureImportanceData = [
    { feature: "Credit Score", importance: 92 },
    { feature: "Account Balance", importance: 85 },
    { feature: "Transaction Freq", importance: 78 },
    { feature: "Age", importance: 65 },
    { feature: "Geography", importance: 58 },
  ];

  const radarData = drivers.map(d => ({
    subject: d.name.split(" ")[0],
    value: d.impact,
  }));

  const handleDownloadAnalysis = () => {
    const csvContent = "data:text/csv;charset=utf-8,Feature,Importance Score\n" +
      featureImportanceData.map(f => `${f.feature},${f.importance}`).join("\n");
    const link = document.createElement("a");
    link.setAttribute("href", encodeURI(csvContent));
    link.setAttribute("download", "feature_analysis.csv");
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-background to-secondary/20">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="mb-8 animate-fade-in">
          <h1 className="text-4xl md:text-5xl font-bold mb-4">Driver Analysis</h1>
          <p className="text-xl text-muted-foreground">
            Key factors influencing risk exposure and their impact assessment
          </p>
        </div>

        {/* Driver Cards */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
          {drivers.map((driver, index) => (
            <Card
              key={index}
              className={`border-2 transition-all duration-300 hover:shadow-elegant hover:-translate-y-1 cursor-pointer animate-scale-in ${
                selectedDriver === driver.name ? "border-primary shadow-elegant" : ""
              }`}
              style={{ animationDelay: `${index * 100}ms` }}
              onClick={() => setSelectedDriver(selectedDriver === driver.name ? null : driver.name)}
            >
              <CardHeader>
                <div className="flex items-center justify-between mb-2">
                  <CardTitle className="text-xl flex items-center gap-2">
                    <driver.icon className="w-5 h-5 text-primary" />
                    {driver.name}
                  </CardTitle>
                  <Badge
                    variant={driver.trend === "increasing" ? "destructive" : "secondary"}
                  >
                    {driver.trend}
                  </Badge>
                </div>
                <CardDescription>Impact: {impactWeights[driver.name]}%</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div>
                    <div className="flex justify-between text-sm mb-2">
                      <span className="text-muted-foreground">Impact Weight</span>
                      <span className="font-medium">{impactWeights[driver.name]}%</span>
                    </div>
                    <Slider
                      value={[impactWeights[driver.name]]}
                      onValueChange={(value) =>
                        setImpactWeights({ ...impactWeights, [driver.name]: value[0] })
                      }
                      max={100}
                      step={1}
                      className="mb-4"
                    />
                  </div>

                  {selectedDriver === driver.name && (
                    <div className="animate-fade-in pt-4 border-t">
                      <p className="text-sm font-medium mb-3">Key Contributing Factors:</p>
                      <div className="flex flex-wrap gap-2">
                        {driver.factors.map((factor, i) => (
                          <Badge key={i} variant="outline" className="text-xs">
                            {factor}
                          </Badge>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>
          ))}
        </div>

        {/* Driver Category Details with Charts */}
        <Card className="border-2 shadow-elegant mb-8">
          <CardHeader>
            <CardTitle>Driver Category Analysis</CardTitle>
            <CardDescription>Visual representation of driver impacts</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={drivers.map(d => ({ name: d.name.split(" ")[0], impact: d.impact }))}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="impact" fill="hsl(var(--primary))" name="Impact Score" />
                </BarChart>
              </ResponsiveContainer>
              <ResponsiveContainer width="100%" height={300}>
                <RadarChart data={radarData}>
                  <PolarGrid />
                  <PolarAngleAxis dataKey="subject" />
                  <PolarRadiusAxis />
                  <Radar name="Impact" dataKey="value" stroke="hsl(var(--primary))" fill="hsl(var(--primary))" fillOpacity={0.6} />
                  <Tooltip />
                </RadarChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        {/* Detailed Feature Breakdown */}
        <Card className="border-2 shadow-elegant mb-8">
          <CardHeader>
            <div className="flex items-center justify-between">
              <div>
                <CardTitle>Detailed Feature Breakdown</CardTitle>
                <CardDescription>Feature importance analysis for churn prediction</CardDescription>
              </div>
              <Button onClick={handleDownloadAnalysis} variant="outline" className="gap-2">
                <Download className="w-4 h-4" />
                Download Feature Analysis
              </Button>
            </div>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={featureImportanceData} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" />
                <YAxis dataKey="feature" type="category" width={120} />
                <Tooltip />
                <Bar dataKey="importance" fill="hsl(var(--primary))" name="Importance Score" />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        {/* Recommended Action Plan */}
        <Card className="border-2 shadow-elegant mb-8">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <CheckCircle2 className="w-5 h-5 text-green-500" />
              Recommended Action Plan
            </CardTitle>
            <CardDescription>Strategic actions to reduce churn risk</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {[
                {
                  priority: "High",
                  action: "Implement proactive customer engagement for high-risk accounts",
                  timeline: "Immediate",
                  impact: "Reduce churn by 25%",
                },
                {
                  priority: "High",
                  action: "Enhance credit scoring model with additional behavioral data",
                  timeline: "2-4 weeks",
                  impact: "Improve prediction accuracy by 15%",
                },
                {
                  priority: "Medium",
                  action: "Launch targeted retention campaigns for medium-risk segments",
                  timeline: "1 month",
                  impact: "Reduce churn by 10%",
                },
                {
                  priority: "Medium",
                  action: "Optimize account balance monitoring thresholds",
                  timeline: "2 months",
                  impact: "Early warning for 30% more at-risk customers",
                },
              ].map((item, index) => (
                <div
                  key={index}
                  className="p-4 border-2 rounded-lg hover:border-primary/50 transition-all"
                >
                  <div className="flex items-start justify-between mb-2">
                    <Badge variant={item.priority === "High" ? "destructive" : "secondary"}>
                      {item.priority} Priority
                    </Badge>
                    <span className="text-xs text-muted-foreground">{item.timeline}</span>
                  </div>
                  <p className="font-semibold mb-2">{item.action}</p>
                  <p className="text-sm text-muted-foreground">Expected Impact: {item.impact}</p>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Correlation Matrix */}
        <Card className="border-2 shadow-elegant">
          <CardHeader>
            <CardTitle>Driver Correlation Matrix</CardTitle>
            <CardDescription>
              Relationship strength between different risk drivers
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr>
                    <th className="p-2 text-left text-sm font-medium text-muted-foreground"></th>
                    {drivers.map((driver) => (
                      <th key={driver.name} className="p-2 text-center text-sm font-medium">
                        {driver.name.split(" ")[0]}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {drivers.map((rowDriver, i) => (
                    <tr key={i} className="border-t">
                      <td className="p-2 text-sm font-medium">{rowDriver.name.split(" ")[0]}</td>
                      {drivers.map((colDriver, j) => {
                        const correlation = i === j ? 100 : Math.abs(i - j) * 15 + 40;
                        return (
                          <td key={j} className="p-2 text-center">
                            <div
                              className={`inline-flex items-center justify-center w-12 h-12 rounded-lg font-semibold text-sm ${
                                correlation > 70
                                  ? "bg-green-500/20 text-green-700 dark:text-green-400"
                                  : correlation > 50
                                  ? "bg-yellow-500/20 text-yellow-700 dark:text-yellow-400"
                                  : "bg-red-500/20 text-red-700 dark:text-red-400"
                              }`}
                            >
                              {correlation}
                            </div>
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default DriverAnalysis;
