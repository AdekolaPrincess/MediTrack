import React from "react";
import { Link } from "react-router";

export default function Menu() {
  return (
    <div className="flex flex-col gap-2 bg-white items-center rounded-xl -mt-80 ml-10 p-5">
      <Link to="/"> Home</Link>
      <Link to="/analytics">Analytics</Link>
    </div>
  );
}
