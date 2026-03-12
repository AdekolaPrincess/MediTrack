import { Route, Routes } from "react-router";
import Navbar from "./components/Navbar";
import Dashboard from "./pages/Dashboard";
import ChatbotPage from "./pages/ChatbotPage";
import AnalyticsPage from "./pages/AnalyticsPage";
import MenuModal from "./components/MenuModal";
import Menu from "./components/Menu";
import { useState } from "react";

function App() {
  const [open, setOpen] = useState(false);
  return (
    <div className="relative flex justify-center items-center h-screen">
      <div className="bg-blue-100/30  w-94 h-167">
        <Navbar />

        <MenuModal open={open} onClose={() => setOpen(false)}>
          <Menu setOpen={setOpen} />
        </MenuModal>

        <Routes>
          <Route path="/" element={<Dashboard setOpen={setOpen} />} />
          <Route path="/chatbot" element={<ChatbotPage setOpen={setOpen} />} />
          <Route
            path="/analytics"
            element={<AnalyticsPage setOpen={setOpen} />}
          />
        </Routes>
      </div>
    </div>
  );
}

export default App;
