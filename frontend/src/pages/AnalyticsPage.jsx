import { Menu } from "lucide-react";
import React from "react";
import Header from "../components/Header";

export default function AnalyticsPage({ setOpen }) {
  return (
    <div className="px-5">
      <Header title="Analytics" setOpen={setOpen} />

      <h2 className="text-2xl font-bold mt-4">Weekly Medication Report</h2>

      <div className="flex flex-col bg-white shadow-lg rounded-xl p-4 mt-4">
        <span className="text-xl font-semibold">Weekly Compliance:</span>
        <span className="text-3xl font-bold text-blue-500">85%</span>
      </div>

      <div className="flex flex-col items-center bg-white shadow-lg rounded-xl p-4 mt-4">
        <span className="text-sm font-semibold text-center">
          Doses Taken This Week
        </span>

        <div className="flex items-end max-w-3xl mx-auto mt-4 gap-10">
          <div className="flex flex-col gap-3  ">
            <span className="flex justify-center items-center bg-blue-500 w-5 h-5 text-white text-xs rounded-lg ml-7 ">
              6
            </span>
            <div className="bg-blue-500 w-12 h-35 rounded-t-lg "></div>
          </div>

          <div className="flex flex-col gap-3  ">
            <span className="flex justify-center items-center bg-blue-500 w-5 h-5 text-white text-xs rounded-lg ml-7 ">
              5
            </span>
            <div className="bg-blue-500 w-12 h-25 rounded-t-lg "></div>
          </div>
        </div>
        <hr className="border-0.5 w-50 opacity-30" />

        <div className="flex opacity-60 text-sm gap-18">
          <span>AM</span>
          <span>PM</span>
        </div>
      </div>

      <div className="flex flex-col bg-white shadow-lg rounded-xl p-4 mt-6">
        <p className="text-xs opacity-50">
          Report summary: Doses were taken on time more times this week than
          lastweek
        </p>
      </div>
    </div>
  );
}
