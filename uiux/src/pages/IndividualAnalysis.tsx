import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Search, User, TrendingUp, AlertCircle } from "lucide-react";
import { useState } from "react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from "recharts";

const individuals = [
  {
    id: 1,
    customerId: "CUST001",
    name: "Sarah Johnson",
    role: "Risk Manager",
    department: "Finance",
    riskScore: 42,
    activeRisks: 12,
    mitigatedRisks: 28,
    status: "low",
  },
  {
    id: 2,
    customerId: "CUST002",
    name: "Michael Chen",
    role: "Compliance Officer",
    department: "Legal",
    riskScore: 78,
    activeRisks: 8,
    mitigatedRisks: 15,
    status: "high",
  },
  {
    id: 3,
    customerId: "CUST003",
    name: "Emily Rodriguez",
    role: "Operations Lead",
    department: "Operations",
    riskScore: 65,
    activeRisks: 15,
    mitigatedRisks: 22,
    status: "medium",
  },
  {
    id: 4,
    customerId: "CUST004",
    name: "David Kim",
    role: "IT Security",
    department: "Technology",
    riskScore: 55,
    activeRisks: 10,
    mitigatedRisks: 18,
    status: "medium",
  },
  {
    id: 5,
    customerId: "CUST005",
    name: "Lisa Thompson",
    role: "Strategic Advisor",
    department: "Executive",
    riskScore: 38,
    activeRisks: 6,
    mitigatedRisks: 32,
    status: "low",
  },
];

const IndividualAnalysis = () => {
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedIndividual, setSelectedIndividual] = useState<number | null>(null);
  const [selectedCustomerId, setSelectedCustomerId] = useState("");

  const riskTrendData = [
    { month: "Jan", score: 45 },
    { month: "Feb", score: 52 },
    { month: "Mar", score: 48 },
    { month: "Apr", score: 58 },
    { month: "May", score: 65 },
    { month: "Jun", score: 62 },
  ];

  const filteredIndividuals = individuals.filter(
    (individual) =>
      individual.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      individual.department.toLowerCase().includes(searchQuery.toLowerCase()) ||
      individual.role.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <div className="min-h-screen bg-gradient-to-b from-background to-secondary/20">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="mb-8 animate-fade-in">
          <h1 className="text-4xl md:text-5xl font-bold mb-4">Individual Analysis</h1>
          <p className="text-xl text-muted-foreground">
            Personnel-level risk assessment and performance tracking
          </p>
        </div>

        {/* Search and Selection */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
          <Card className="border-2">
            <CardContent className="pt-6">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-muted-foreground" />
                <Input
                  placeholder="Search by name, role, or department..."
                  className="pl-10"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                />
              </div>
            </CardContent>
          </Card>

          <Card className="border-2">
            <CardContent className="pt-6">
              <Select value={selectedCustomerId} onValueChange={setSelectedCustomerId}>
                <SelectTrigger>
                  <SelectValue placeholder="Select Customer by ID" />
                </SelectTrigger>
                <SelectContent>
                  {individuals.map((individual) => (
                    <SelectItem key={individual.id} value={individual.customerId}>
                      {individual.customerId} - {individual.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              {selectedCustomerId && (
                <div className="mt-3 p-3 bg-secondary/30 rounded-lg">
                  <p className="text-sm font-medium">Customer ID: {selectedCustomerId}</p>
                </div>
              )}
            </CardContent>
          </Card>
        </div>

        {/* Risk Trend Chart */}
        <Card className="border-2 shadow-elegant mb-8">
          <CardHeader>
            <CardTitle>Customer Risk Trend</CardTitle>
            <CardDescription>Risk score progression over time</CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={riskTrendData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="month" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Line 
                  type="monotone" 
                  dataKey="score" 
                  stroke="hsl(var(--primary))" 
                  strokeWidth={2}
                  name="Risk Score"
                />
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        {/* Individual Cards Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
          {filteredIndividuals.map((individual, index) => (
            <Card
              key={individual.id}
              className={`border-2 transition-all duration-300 hover:shadow-elegant hover:-translate-y-1 cursor-pointer animate-scale-in ${
                selectedIndividual === individual.id ? "border-primary shadow-elegant" : ""
              }`}
              style={{ animationDelay: `${index * 100}ms` }}
              onClick={() =>
                setSelectedIndividual(selectedIndividual === individual.id ? null : individual.id)
              }
            >
              <CardHeader>
                <div className="flex items-center gap-4">
                  <Avatar className="w-12 h-12">
                    <AvatarFallback className="bg-gradient-to-br from-primary to-accent text-white font-semibold">
                      {individual.name
                        .split(" ")
                        .map((n) => n[0])
                        .join("")}
                    </AvatarFallback>
                  </Avatar>
                  <div className="flex-1">
                    <CardTitle className="text-lg">{individual.name}</CardTitle>
                    <CardDescription>
                      ID: {individual.customerId} • {individual.role} • {individual.department}
                    </CardDescription>
                  </div>
                  <Badge
                    variant={
                      individual.status === "high"
                        ? "destructive"
                        : individual.status === "medium"
                        ? "secondary"
                        : "outline"
                    }
                    className={
                      individual.status === "low"
                        ? "border-green-500 text-green-700 dark:text-green-400"
                        : ""
                    }
                  >
                    {individual.status}
                  </Badge>
                </div>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div>
                    <div className="flex justify-between text-sm mb-2">
                      <span className="text-muted-foreground">Risk Score</span>
                      <span className="font-medium">{individual.riskScore}/100</span>
                    </div>
                    <div className="h-2 bg-secondary rounded-full overflow-hidden">
                      <div
                        className={`h-full transition-all duration-1000 ${
                          individual.status === "high"
                            ? "bg-gradient-to-r from-destructive to-red-600"
                            : individual.status === "medium"
                            ? "bg-gradient-to-r from-yellow-500 to-orange-500"
                            : "bg-gradient-to-r from-green-500 to-emerald-500"
                        }`}
                        style={{ width: `${individual.riskScore}%` }}
                      />
                    </div>
                  </div>

                  <div className="grid grid-cols-2 gap-4">
                    <div className="p-3 bg-secondary/50 rounded-lg">
                      <div className="flex items-center gap-2 mb-1">
                        <AlertCircle className="w-4 h-4 text-destructive" />
                        <span className="text-xs text-muted-foreground">Active Risks</span>
                      </div>
                      <span className="text-2xl font-bold">{individual.activeRisks}</span>
                    </div>
                    <div className="p-3 bg-secondary/50 rounded-lg">
                      <div className="flex items-center gap-2 mb-1">
                        <TrendingUp className="w-4 h-4 text-green-500" />
                        <span className="text-xs text-muted-foreground">Mitigated</span>
                      </div>
                      <span className="text-2xl font-bold">{individual.mitigatedRisks}</span>
                    </div>
                  </div>

                  {selectedIndividual === individual.id && (
                    <div className="pt-4 border-t animate-fade-in">
                      <p className="text-sm text-muted-foreground">
                        <strong>Performance Summary:</strong> {individual.name} has successfully
                        mitigated {individual.mitigatedRisks} risks with {individual.activeRisks}{" "}
                        currently being monitored. Their risk management efficiency is{" "}
                        {Math.round(
                          (individual.mitigatedRisks /
                            (individual.activeRisks + individual.mitigatedRisks)) *
                            100
                        )}
                        %.
                      </p>
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>
          ))}
        </div>

        {filteredIndividuals.length === 0 && (
          <Card className="border-2">
            <CardContent className="py-12 text-center">
              <User className="w-12 h-12 mx-auto mb-4 text-muted-foreground" />
              <p className="text-lg text-muted-foreground">No individuals found matching your search</p>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );
};

export default IndividualAnalysis;
