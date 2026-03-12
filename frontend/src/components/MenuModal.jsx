import React from "react";

export default function MenuModal({ open, onClose, children }) {
  return (
    <div
      onClick={onClose}
      className={`fixed inset-0 flex items-center mx-auto my-auto w-94 h-167  z-2 ${open ? "visible bg-black/10" : "invisible"}`}
    >
      {children}{" "}
    </div>
  );
}
