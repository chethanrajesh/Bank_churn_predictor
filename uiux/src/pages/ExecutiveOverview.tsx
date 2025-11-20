import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { TrendingUp, TrendingDown, AlertTriangle, CheckCircle2, DollarSign, Users } from "lucide-react";

const kpis = [
  { label: "Total Risk Exposure", value: "$2.4M", change: "+12.5%", trend: "up", icon: DollarSign },
  { label: "Active Risk Items", value: "127", change: "-8.2%", trend: "down", icon: AlertTriangle },
  { label: "Compliance Score", value: "94.2%", change: "+3.1%", trend: "up", icon: CheckCircle2 },
  { label: "Risk Personnel", value: "45", change: "+5", trend: "up", icon: Users },
];

const alerts = [
  { severity: "high", message: "Market volatility increased by 15%", time: "2 hours ago" },
  { severity: "medium", message: "New compliance requirement identified", time: "5 hours ago" },
  { severity: "low", message: "Quarterly risk assessment due next week", time: "1 day ago" },
];

const ExecutiveOverview = () => {
  return (
    <div className="min-h-screen bg-gradient-to-b from-background to-secondary/20">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="mb-8 animate-fade-in">
          <h1 className="text-4xl md:text-5xl font-bold mb-4">Executive Overview</h1>
          <p className="text-xl text-muted-foreground">
            High-level insights and key performance indicators
          </p>
        </div>

        {/* KPI Cards */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          {kpis.map((kpi, index) => (
            <Card
              key={index}
              className="hover:shadow-elegant transition-all duration-300 hover:-translate-y-1 border-2 animate-scale-in"
              style={{ animationDelay: `${index * 100}ms` }}
            >
              <CardHeader className="flex flex-row items-center justify-between pb-2">
                <CardTitle className="text-sm font-medium text-muted-foreground">
                  {kpi.label}
                </CardTitle>
                <kpi.icon className="w-4 h-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-3xl font-bold mb-1">{kpi.value}</div>
                <div className="flex items-center gap-1 text-sm">
                  {kpi.trend === "up" ? (
                    <TrendingUp className="w-4 h-4 text-green-500" />
                  ) : (
                    <TrendingDown className="w-4 h-4 text-destructive" />
                  )}
                  <span className={kpi.trend === "up" ? "text-green-500" : "text-destructive"}>
                    {kpi.change}
                  </span>
                  <span className="text-muted-foreground">vs last month</span>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Risk Distribution */}
          <Card className="border-2 shadow-elegant">
            <CardHeader>
              <CardTitle>Risk Distribution by Category</CardTitle>
              <CardDescription>Breakdown of risk exposure across departments</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {[
                { name: "Financial", percentage: 35, color: "bg-blue-500" },
                { name: "Operational", percentage: 28, color: "bg-purple-500" },
                { name: "Strategic", percentage: 22, color: "bg-orange-500" },
                { name: "Compliance", percentage: 15, color: "bg-green-500" },
              ].map((item, index) => (
                <div key={index} className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span className="font-medium">{item.name}</span>
                    <span className="text-muted-foreground">{item.percentage}%</span>
                  </div>
                  <div className="h-2 bg-secondary rounded-full overflow-hidden">
                    <div
                      className={`h-full ${item.color} transition-all duration-1000`}
                      style={{ width: `${item.percentage}%`, animationDelay: `${index * 100}ms` }}
                    />
                  </div>
                </div>
              ))}
            </CardContent>
          </Card>

          {/* Recent Alerts */}
          <Card className="border-2 shadow-elegant">
            <CardHeader>
              <CardTitle>Recent Alerts</CardTitle>
              <CardDescription>Latest risk notifications and updates</CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
              {alerts.map((alert, index) => (
                <div
                  key={index}
                  className="flex items-start gap-3 p-3 rounded-lg border-2 hover:border-primary/50 transition-all cursor-pointer"
                >
                  <Badge
                    variant={alert.severity === "high" ? "destructive" : "secondary"}
                    className="mt-0.5"
                  >
                    {alert.severity}
                  </Badge>
                  <div className="flex-1">
                    <p className="text-sm font-medium">{alert.message}</p>
                    <p className="text-xs text-muted-foreground mt-1">{alert.time}</p>
                  </div>
                </div>
              ))}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
};

export default ExecutiveOverview;
