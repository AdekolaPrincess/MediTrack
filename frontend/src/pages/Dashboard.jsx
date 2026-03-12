import { BotMessageSquare, CircleX, Clock4 } from "lucide-react";
import React from "react";
import { Link } from "react-router";
import Header from "../components/Header";

export default function Dashboard({ setOpen }) {
  return (
    <div className="relative ">
      <Header title="Dashboard" setOpen={setOpen} />

      <h2 className="font-bold text-2xl px-7 mt-7">
        Smart Pill Dispenser <br />
        Dashboard
      </h2>
      <div className="px-7 mt-5">
        <span className="font-bold">Patient Name:</span>
        <span>John Doe</span>
      </div>
      <div className="px-7 mt-2">
        <span className="font-bold">Patient Condition:</span>
        <span>Hypertension</span>
      </div>

      <div className="max-w-4xl mx-auto flex justify-center gap-8  mt-15  ">
        <div
          className=" bg-white rounded-xl flex flex-col items-center px-5 py-2"
          style={{ boxShadow: "0px 4px 10px rgba(0,0,0,0.2)" }}
        >
          <span>Morning Dose</span>
          <span className="font-bold">08:00 AM</span>
          <Clock4 className="text-yellow-400 size-7 mt-2" />
          <span className="text-yellow-400 text-sm ">Pending</span>
        </div>
        <div
          className=" bg-white rounded-xl flex flex-col items-center px-8 py-2"
          style={{ boxShadow: "0px 4px 10px rgba(0,0,0,0.2)" }}
        >
          <span>Night Dose</span>
          <span className="font-bold">08:00 PM</span>
          <CircleX className="text-red-600 size-7 mt-2" />
          <span className="text-red-600 text-sm ">Missed</span>
        </div>
      </div>

      <div
        className="bg-white max-w-4xl mx-7 rounded-xl flex flex-col  items-center  px-5 py-2 mt-8"
        style={{ boxShadow: "0px 4px 10px rgba(0,0,0,0.2)" }}
      >
        {/* chatbot button */}
        <Link
          to="/chatbot"
          className="fixed bg-blue-500 w-12 h-12 flex justify-center items-center rounded-full right-2 bottom-40 "
        >
          <BotMessageSquare className="text-white " />
        </Link>

        <div className="flex justify-center gap-14 ">
          <span>Time</span>
          <span>Dose</span>
          <span>Status</span>
        </div>
        <hr className="border border-gray-200 w-full mt-2" />

        <div className="flex justify-center gap-7 mr-3 mt-2">
          <span>08:00 AM</span>
          <span className="mr-6">2 pills</span>
          <span>Taken</span>
        </div>
        <hr className="border border-gray-200 w-full mt-2" />
        <div className="flex justify-center gap-7 mr-3 mt-2">
          <span>08:00 PM</span>
          <span className="mr-6">2 pills</span>
          <span>Taken</span>
        </div>
      </div>
    </div>
  );
}
