import { Menu, SendHorizontal } from "lucide-react";
import React, { useState } from "react";
import Header from "../components/Header";

export default function ChatbotPage({ setOpen }) {
  const [chat, setChat] = useState("");
  return (
    <div className="px-4">
      <Header title="Chatbot" setOpen={setOpen} />

      <h2 className="pl-2 mt-4 font-bold text-xl">AI Assistant Chatbot</h2>

      <div className="border border-blue-500 h-100 w-full"></div>

      <div className="flex justify-center gap-4 items-center mt-8 ">
        <input
          className="border-2 border-blue-500 h-10 w-full rounded-xl p-4"
          type="text"
          placeholder="Type your question"
          value={chat}
          onChange={(e) => setChat(e.target.value)}
        />
        <button className="px-2 py-1.5 bg-blue-500 rounded-xl ">
          <SendHorizontal size={25} color="white" />
        </button>
      </div>
    </div>
  );
}
