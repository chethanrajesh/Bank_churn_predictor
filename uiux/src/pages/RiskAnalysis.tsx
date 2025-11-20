import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Slider } from "@/components/ui/slider";
import { Button } from "@/components/ui/button";
import { AlertTriangle, Shield, Download } from "lucide-react";
import { useState } from "react";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell, Legend } from "recharts";

const riskCategories = [
  {
    id: "market",
    name: "Market Risk",
    level: 78,
    items: [
      { name: "Currency Fluctuation", severity: "high", probability: 85, impact: 72 },
      { name: "Interest Rate Changes", severity: "medium", probability: 65, impact: 58 },
      { name: "Commodity Price Volatility", severity: "high", probability: 78, impact: 80 },
    ],
  },
  {
    id: "operational",
    name: "Operational Risk",
    level: 62,
    items: [
      { name: "System Downtime", severity: "high", probability: 45, impact: 85 },
      { name: "Supply Chain Disruption", severity: "medium", probability: 60, impact: 65 },
      { name: "Process Inefficiencies", severity: "low", probability: 70, impact: 40 },
    ],
  },
  {
    id: "compliance",
    name: "Compliance Risk",
    level: 45,
    items: [
      { name: "Regulatory Changes", severity: "medium", probability: 55, impact: 70 },
      { name: "Data Privacy Violations", severity: "high", probability: 30, impact: 90 },
      { name: "Audit Findings", severity: "low", probability: 40, impact: 45 },
    ],
  },
];

const RiskAnalysis = () => {
  const [selectedCategory, setSelectedCategory] = useState(riskCategories[0].id);
  const [sampleSize, setSampleSize] = useState([100]);
  const [topFeatures, setTopFeatures] = useState([10]);

  const chartData = [
    { name: "High Risk", value: 45, color: "hsl(var(--destructive))" },
    { name: "Medium Risk", value: 30, color: "hsl(45, 93%, 47%)" },
    { name: "Low Risk", value: 25, color: "hsl(142, 76%, 36%)" },
  ];

  const distributionData = [
    { category: "Credit Score", high: 35, medium: 25, low: 15 },
    { category: "Account Age", high: 28, medium: 32, low: 18 },
    { category: "Transaction Volume", high: 42, medium: 28, low: 12 },
    { category: "Balance Trend", high: 38, medium: 30, low: 20 },
  ];

  const handleExportHighRisk = () => {
    const csvContent = "data:text/csv;charset=utf-8,Customer ID,Risk Score,Category\n" +
      "CUST001,92,High Risk\nCUST005,88,High Risk\nCUST012,85,High Risk";
    const link = document.createElement("a");
    link.setAttribute("href", encodeURI(csvContent));
    link.setAttribute("download", "high_risk_customers.csv");
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-background to-secondary/20">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="mb-8 animate-fade-in">
          <h1 className="text-4xl md:text-5xl font-bold mb-4">Risk Analysis</h1>
          <p className="text-xl text-muted-foreground">
            Comprehensive assessment of risk factors and mitigation strategies
          </p>
        </div>

        {/* Risk Overview Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
          {riskCategories.map((category, index) => (
            <Card
              key={category.id}
              className={`cursor-pointer border-2 transition-all duration-300 hover:shadow-elegant hover:-translate-y-1 animate-scale-in ${
                selectedCategory === category.id ? "border-primary shadow-elegant" : ""
              }`}
              style={{ animationDelay: `${index * 100}ms` }}
              onClick={() => setSelectedCategory(category.id)}
            >
              <CardHeader>
                <div className="flex items-center justify-between">
                  <CardTitle className="text-lg">{category.name}</CardTitle>
                  <Shield className="w-5 h-5 text-primary" />
                </div>
              </CardHeader>
              <CardContent>
                <div className="flex items-end gap-2 mb-2">
                  <span className="text-3xl font-bold">{category.level}</span>
                  <span className="text-muted-foreground mb-1">/ 100</span>
                </div>
                <div className="h-2 bg-secondary rounded-full overflow-hidden">
                  <div
                    className={`h-full transition-all duration-1000 ${
                      category.level > 70
                        ? "bg-gradient-to-r from-destructive to-red-600"
                        : category.level > 50
                        ? "bg-gradient-to-r from-yellow-500 to-orange-500"
                        : "bg-gradient-to-r from-green-500 to-emerald-500"
                    }`}
                    style={{ width: `${category.level}%` }}
                  />
                </div>
              </CardContent>
            </Card>
          ))}
        </div>

        {/* High Risk Distribution */}
        <Card className="border-2 shadow-elegant mb-8">
          <CardHeader>
            <div className="flex items-center justify-between">
              <div>
                <CardTitle>High Risk Distribution</CardTitle>
                <CardDescription>Customer risk level breakdown</CardDescription>
              </div>
              <Button onClick={handleExportHighRisk} variant="outline" className="gap-2">
                <Download className="w-4 h-4" />
                Export High Risk List
              </Button>
            </div>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <ResponsiveContainer width="100%" height={300}>
                <PieChart>
                  <Pie
                    data={chartData}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                    outerRadius={100}
                    fill="hsl(var(--primary))"
                    dataKey="value"
                  >
                    {chartData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip />
                  <Legend />
                </PieChart>
              </ResponsiveContainer>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={distributionData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="category" angle={-45} textAnchor="end" height={100} />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="high" fill="hsl(var(--destructive))" name="High Risk" />
                  <Bar dataKey="medium" fill="hsl(45, 93%, 47%)" name="Medium Risk" />
                  <Bar dataKey="low" fill="hsl(142, 76%, 36%)" name="Low Risk" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        {/* Visualization Controls */}
        <Card className="border-2 shadow-elegant mb-8">
          <CardHeader>
            <CardTitle>Visualization Settings</CardTitle>
            <CardDescription>Adjust sample size and feature display</CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            <div>
              <div className="flex justify-between mb-3">
                <span className="text-sm font-medium">Sample Size</span>
                <span className="text-sm text-muted-foreground">{sampleSize[0]} customers</span>
              </div>
              <Slider
                value={sampleSize}
                onValueChange={setSampleSize}
                max={500}
                min={10}
                step={10}
              />
            </div>
            <div>
              <div className="flex justify-between mb-3">
                <span className="text-sm font-medium">Number of Top Features to Show</span>
                <span className="text-sm text-muted-foreground">{topFeatures[0]} features</span>
              </div>
              <Slider
                value={topFeatures}
                onValueChange={setTopFeatures}
                max={20}
                min={5}
                step={1}
              />
            </div>
          </CardContent>
        </Card>

        {/* Detailed Risk Items */}
        <Card className="border-2 shadow-elegant">
          <CardHeader>
            <CardTitle>Risk Item Details</CardTitle>
            <CardDescription>Individual risk factors and their assessment metrics</CardDescription>
          </CardHeader>
          <CardContent>
            <Tabs value={selectedCategory} onValueChange={setSelectedCategory}>
              <TabsList className="grid w-full grid-cols-3 mb-6">
                {riskCategories.map((category) => (
                  <TabsTrigger key={category.id} value={category.id}>
                    {category.name}
                  </TabsTrigger>
                ))}
              </TabsList>

              {riskCategories.map((category) => (
                <TabsContent key={category.id} value={category.id} className="space-y-4">
                  {category.items.map((item, index) => (
                    <div
                      key={index}
                      className="p-4 border-2 rounded-lg hover:border-primary/50 transition-all animate-fade-in"
                      style={{ animationDelay: `${index * 100}ms` }}
                    >
                      <div className="flex items-start justify-between mb-3">
                        <div className="flex items-center gap-3">
                          <AlertTriangle
                            className={`w-5 h-5 ${
                              item.severity === "high"
                                ? "text-destructive"
                                : item.severity === "medium"
                                ? "text-yellow-500"
                                : "text-green-500"
                            }`}
                          />
                          <span className="font-semibold">{item.name}</span>
                        </div>
                        <Badge
                          variant={item.severity === "high" ? "destructive" : "secondary"}
                        >
                          {item.severity}
                        </Badge>
                      </div>

                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div>
                          <div className="flex justify-between text-sm mb-2">
                            <span className="text-muted-foreground">Probability</span>
                            <span className="font-medium">{item.probability}%</span>
                          </div>
                          <div className="h-2 bg-secondary rounded-full overflow-hidden">
                            <div
                              className="h-full bg-gradient-to-r from-blue-500 to-cyan-500 transition-all duration-1000"
                              style={{ width: `${item.probability}%` }}
                            />
                          </div>
                        </div>

                        <div>
                          <div className="flex justify-between text-sm mb-2">
                            <span className="text-muted-foreground">Impact</span>
                            <span className="font-medium">{item.impact}%</span>
                          </div>
                          <div className="h-2 bg-secondary rounded-full overflow-hidden">
                            <div
                              className="h-full bg-gradient-to-r from-purple-500 to-pink-500 transition-all duration-1000"
                              style={{ width: `${item.impact}%` }}
                            />
                          </div>
                        </div>
                      </div>
                    </div>
                  ))}
                </TabsContent>
              ))}
            </Tabs>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default RiskAnalysis;
