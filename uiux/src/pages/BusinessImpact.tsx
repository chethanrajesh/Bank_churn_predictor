import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { DollarSign, TrendingDown, Users, AlertTriangle, Building2, Target, TrendingUp, CheckCircle2 } from "lucide-react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend, AreaChart, Area } from "recharts";

const impactMetrics = [
  {
    label: "Potential Revenue Loss",
    value: "$8.2M",
    change: "+15.3%",
    icon: DollarSign,
    trend: "negative",
  },
  {
    label: "Operational Disruption",
    value: "142 hrs",
    change: "+22.1%",
    icon: TrendingDown,
    trend: "negative",
  },
  {
    label: "Affected Personnel",
    value: "1,250",
    change: "+8.5%",
    icon: Users,
    trend: "negative",
  },
  {
    label: "Critical Incidents",
    value: "23",
    change: "-12.3%",
    icon: AlertTriangle,
    trend: "positive",
  },
];

const businessUnits = [
  { name: "Sales & Marketing", impact: 85, revenue: "$3.2M", incidents: 8 },
  { name: "Operations", impact: 72, revenue: "$2.1M", incidents: 12 },
  { name: "Technology", impact: 68, revenue: "$1.8M", incidents: 6 },
  { name: "Finance", impact: 55, revenue: "$1.1M", incidents: 4 },
];

const scenarios = [
  {
    name: "Major System Outage",
    probability: 15,
    impact: "$5.2M",
    duration: "48 hours",
    severity: "critical",
  },
  {
    name: "Data Breach",
    probability: 8,
    impact: "$12.5M",
    duration: "2 weeks",
    severity: "critical",
  },
  {
    name: "Supply Chain Disruption",
    probability: 35,
    impact: "$2.8M",
    duration: "1 week",
    severity: "high",
  },
  {
    name: "Compliance Violation",
    probability: 12,
    impact: "$4.1M",
    duration: "3 weeks",
    severity: "high",
  },
];

const BusinessImpact = () => {
  const financialData = [
    { month: "Jan", revenue: 4.2, cost: 1.8, roi: 2.4 },
    { month: "Feb", revenue: 4.5, cost: 1.9, roi: 2.6 },
    { month: "Mar", revenue: 4.8, cost: 2.1, roi: 2.7 },
    { month: "Apr", revenue: 5.2, cost: 2.3, roi: 2.9 },
    { month: "May", revenue: 5.6, cost: 2.4, roi: 3.2 },
    { month: "Jun", revenue: 6.0, cost: 2.5, roi: 3.5 },
  ];

  const projectionData = [
    { quarter: "Q1 2025", baseline: 8.2, optimized: 6.1, savings: 2.1 },
    { quarter: "Q2 2025", baseline: 8.8, optimized: 5.8, savings: 3.0 },
    { quarter: "Q3 2025", baseline: 9.2, optimized: 5.4, savings: 3.8 },
    { quarter: "Q4 2025", baseline: 9.6, optimized: 5.0, savings: 4.6 },
  ];

  return (
    <div className="min-h-screen bg-gradient-to-b from-background to-secondary/20">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="mb-8 animate-fade-in">
          <h1 className="text-4xl md:text-5xl font-bold mb-4">Business Impact</h1>
          <p className="text-xl text-muted-foreground">
            Financial and operational impact assessment of identified risks
          </p>
        </div>

        {/* Impact Metrics */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          {impactMetrics.map((metric, index) => (
            <Card
              key={index}
              className="border-2 hover:shadow-elegant transition-all duration-300 hover:-translate-y-1 animate-scale-in"
              style={{ animationDelay: `${index * 100}ms` }}
            >
              <CardHeader className="flex flex-row items-center justify-between pb-2">
                <CardTitle className="text-sm font-medium text-muted-foreground">
                  {metric.label}
                </CardTitle>
                <metric.icon
                  className={`w-4 h-4 ${
                    metric.trend === "negative" ? "text-destructive" : "text-green-500"
                  }`}
                />
              </CardHeader>
              <CardContent>
                <div className="text-3xl font-bold mb-1">{metric.value}</div>
                <div className="flex items-center gap-1 text-sm">
                  <span
                    className={metric.trend === "negative" ? "text-destructive" : "text-green-500"}
                  >
                    {metric.change}
                  </span>
                  <span className="text-muted-foreground">from last quarter</span>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
          {/* Business Units Impact */}
          <Card className="border-2 shadow-elegant">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Building2 className="w-5 h-5 text-primary" />
                Impact by Business Unit
              </CardTitle>
              <CardDescription>Risk exposure across departments</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {businessUnits.map((unit, index) => (
                <div key={index} className="space-y-2 animate-fade-in" style={{ animationDelay: `${index * 100}ms` }}>
                  <div className="flex justify-between items-center">
                    <span className="font-medium">{unit.name}</span>
                    <div className="flex items-center gap-3">
                      <span className="text-sm text-muted-foreground">{unit.revenue} at risk</span>
                      <Badge variant={unit.impact > 70 ? "destructive" : "secondary"}>
                        {unit.incidents} incidents
                      </Badge>
                    </div>
                  </div>
                  <div className="h-2 bg-secondary rounded-full overflow-hidden">
                    <div
                      className={`h-full transition-all duration-1000 ${
                        unit.impact > 70
                          ? "bg-gradient-to-r from-destructive to-red-600"
                          : "bg-gradient-to-r from-yellow-500 to-orange-500"
                      }`}
                      style={{ width: `${unit.impact}%` }}
                    />
                  </div>
                </div>
              ))}
            </CardContent>
          </Card>

          {/* Scenario Analysis */}
          <Card className="border-2 shadow-elegant">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Target className="w-5 h-5 text-primary" />
                Worst-Case Scenarios
              </CardTitle>
              <CardDescription>High-impact risk scenarios and potential losses</CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
              {scenarios.map((scenario, index) => (
                <div
                  key={index}
                  className="p-4 border-2 rounded-lg hover:border-primary/50 transition-all animate-fade-in"
                  style={{ animationDelay: `${index * 100}ms` }}
                >
                  <div className="flex items-start justify-between mb-2">
                    <div className="flex-1">
                      <div className="flex items-center gap-2 mb-1">
                        <span className="font-semibold text-sm">{scenario.name}</span>
                        <Badge
                          variant={scenario.severity === "critical" ? "destructive" : "secondary"}
                          className="text-xs"
                        >
                          {scenario.severity}
                        </Badge>
                      </div>
                      <div className="text-xs text-muted-foreground space-y-1">
                        <div>Probability: {scenario.probability}%</div>
                        <div>Estimated Impact: {scenario.impact}</div>
                        <div>Recovery Time: {scenario.duration}</div>
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </CardContent>
          </Card>
        </div>

        {/* Financial Analysis */}
        <Card className="border-2 shadow-elegant mb-8">
          <CardHeader>
            <CardTitle>Financial Analysis & Performance</CardTitle>
            <CardDescription>Revenue, costs, and ROI trends</CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={financialData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="month" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="revenue" stroke="hsl(142, 76%, 36%)" strokeWidth={2} name="Revenue ($M)" />
                <Line type="monotone" dataKey="cost" stroke="hsl(var(--destructive))" strokeWidth={2} name="Cost ($M)" />
                <Line type="monotone" dataKey="roi" stroke="hsl(var(--primary))" strokeWidth={2} name="ROI ($M)" />
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        {/* ROI Projections */}
        <Card className="border-2 shadow-elegant mb-8">
          <CardHeader>
            <CardTitle>ROI Projections with Optimization</CardTitle>
            <CardDescription>Projected savings from implementing churn reduction strategies</CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <AreaChart data={projectionData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="quarter" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Area type="monotone" dataKey="baseline" stackId="1" stroke="hsl(var(--destructive))" fill="hsl(var(--destructive))" fillOpacity={0.6} name="Baseline Loss ($M)" />
                <Area type="monotone" dataKey="optimized" stackId="2" stroke="hsl(45, 93%, 47%)" fill="hsl(45, 93%, 47%)" fillOpacity={0.6} name="With Optimization ($M)" />
              </AreaChart>
            </ResponsiveContainer>
            <div className="mt-6 grid grid-cols-1 md:grid-cols-3 gap-4">
              {projectionData.map((item, index) => (
                <div key={index} className="p-4 bg-green-500/10 border-2 border-green-500/20 rounded-lg">
                  <p className="text-sm text-muted-foreground mb-1">{item.quarter}</p>
                  <p className="text-2xl font-bold text-green-600">${item.savings}M</p>
                  <p className="text-xs text-muted-foreground">Projected Savings</p>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Recommended Actions */}
        <Card className="border-2 shadow-elegant mb-8">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <CheckCircle2 className="w-5 h-5 text-green-500" />
              Recommended Actions
            </CardTitle>
            <CardDescription>Strategic initiatives to maximize business value</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {[
                {
                  category: "Customer Retention",
                  action: "Launch personalized retention program for top-tier customers",
                  investment: "$250K",
                  expectedReturn: "$2.1M",
                  timeframe: "6 months",
                  priority: "High",
                },
                {
                  category: "Risk Management",
                  action: "Deploy predictive analytics for early churn detection",
                  investment: "$180K",
                  expectedReturn: "$1.5M",
                  timeframe: "4 months",
                  priority: "High",
                },
                {
                  category: "Product Enhancement",
                  action: "Develop premium features based on at-risk customer feedback",
                  investment: "$320K",
                  expectedReturn: "$2.8M",
                  timeframe: "8 months",
                  priority: "Medium",
                },
                {
                  category: "Customer Experience",
                  action: "Implement AI-powered customer support for 24/7 assistance",
                  investment: "$150K",
                  expectedReturn: "$1.2M",
                  timeframe: "3 months",
                  priority: "Medium",
                },
              ].map((item, index) => (
                <div
                  key={index}
                  className="p-5 border-2 rounded-lg hover:border-primary/50 transition-all"
                >
                  <div className="flex items-start justify-between mb-3">
                    <div>
                      <Badge variant={item.priority === "High" ? "destructive" : "secondary"} className="mb-2">
                        {item.priority} Priority
                      </Badge>
                      <p className="text-sm text-muted-foreground">{item.category}</p>
                    </div>
                    <div className="text-right">
                      <p className="text-sm text-muted-foreground">ROI</p>
                      <p className="text-lg font-bold text-green-600">
                        {((parseFloat(item.expectedReturn.replace(/[$M]/g, "")) / parseFloat(item.investment.replace(/[$K]/g, "")) * 1000) - 1).toFixed(1)}x
                      </p>
                    </div>
                  </div>
                  <h4 className="font-semibold mb-2">{item.action}</h4>
                  <div className="grid grid-cols-3 gap-4 text-sm">
                    <div>
                      <p className="text-muted-foreground">Investment</p>
                      <p className="font-medium">{item.investment}</p>
                    </div>
                    <div>
                      <p className="text-muted-foreground">Expected Return</p>
                      <p className="font-medium text-green-600">{item.expectedReturn}</p>
                    </div>
                    <div>
                      <p className="text-muted-foreground">Timeframe</p>
                      <p className="font-medium">{item.timeframe}</p>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Financial Projection Summary */}
        <Card className="border-2 shadow-elegant">
          <CardHeader>
            <CardTitle>Financial Impact Projection</CardTitle>
            <CardDescription>12-month risk cost forecast</CardDescription>
          </CardHeader>
          <CardContent>
            <Tabs defaultValue="quarterly">
              <TabsList className="grid w-full grid-cols-2 mb-6">
                <TabsTrigger value="quarterly">Quarterly</TabsTrigger>
                <TabsTrigger value="annual">Annual</TabsTrigger>
              </TabsList>
              <TabsContent value="quarterly" className="space-y-4">
                {["Q1 2025", "Q2 2025", "Q3 2025", "Q4 2025"].map((quarter, index) => {
                  const cost = 1.8 + index * 0.3;
                  return (
                    <div key={index} className="flex items-center justify-between p-3 bg-secondary/30 rounded-lg">
                      <span className="font-medium">{quarter}</span>
                      <span className="text-lg font-bold text-destructive">${cost.toFixed(1)}M</span>
                    </div>
                  );
                })}
              </TabsContent>
              <TabsContent value="annual" className="space-y-4">
                <div className="p-6 bg-gradient-to-r from-destructive/10 to-red-600/10 rounded-lg border-2 border-destructive/20">
                  <div className="text-center">
                    <p className="text-sm text-muted-foreground mb-2">Total Projected Impact</p>
                    <p className="text-5xl font-bold text-destructive mb-2">$9.6M</p>
                    <p className="text-sm text-muted-foreground">Based on current risk trends</p>
                  </div>
                </div>
              </TabsContent>
            </Tabs>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default BusinessImpact;
