import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

console.log("API BASE FROM ENV =", import.meta.env.VITE_API_BASE);
export const API_BASE = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000";

