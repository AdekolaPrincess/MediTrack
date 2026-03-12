import { Menu } from "lucide-react";
import React from "react";

export default function Header({ title, setOpen }) {
  return (
    <div>
      <div className=" w-80 mx-auto bg-blue-500 flex text-white px-3 py-3 rounded-xl gap-5 mt-5">
        <Menu onClick={() => setOpen(true)} className="text-white" />
        <span>{title}</span>
      </div>
    </div>
  );
}
