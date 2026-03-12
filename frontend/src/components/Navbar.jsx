import React from "react";
import { useNavigate } from "react-router";

export default function Navbar() {
  const navigate = useNavigate();
  return (
    <div className="max-w-4xl flex justify-center">
      <h1
        onClick={() => navigate("/")}
        className="mt-5 text-2xl font-bold text-blue-500"
      >
        MEDITRACK
      </h1>
    </div>
  );
}
