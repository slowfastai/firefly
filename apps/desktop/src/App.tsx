import Logo from "@/components/Logo";
import QuickActions from "@/components/QuickActions";
import SearchBox from "@/components/SearchBox";
import "./App.css";

const quickActionItems = [
  { label: "Fact Check" },
  { label: "Summarize" },
  { label: "Analyze" },
  { label: "Plan" },
  { label: "Recommend" },
  { label: "Inspiration" },
];

function App() {
  return (
    <div className="app-container">
      <Logo />
      <SearchBox />
      <QuickActions items={quickActionItems} />
    </div>
  );
}

export default App;
