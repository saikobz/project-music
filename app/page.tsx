import Image from "next/image";
import UploadBox from "./components/UploadBox";

export default function Home() {
  return (
    <main className="min-h-screen bg-gray-900 text-white p-4">
      <UploadBox />
    </main>
  );
}
