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
    <>
      <div className="bg-blue-100/30 h-screen">
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
    </>
  );
}

export default App;
