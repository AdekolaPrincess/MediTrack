import React from "react";

export default function MenuModal({ open, onClose, children }) {
  return (
    <div
      onClick={onClose}
      className={`fixed inset-0 flex items-center r z-2  ${open ? "visible bg-black/10" : "invisible"}`}
    >
      {children}{" "}
    </div>
  );
}
