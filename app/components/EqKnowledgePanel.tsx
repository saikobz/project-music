"use client";

import React, { useId, useState } from "react";

type EqGuideLink = {
  title: string;
  source: string;
  url: string;
  tag: "พื้นฐาน" | "มิกซ์" | "โทนเสียง";
};

const EQ_GUIDE_LINKS: EqGuideLink[] = [
  {
    title: "EQ Fundamentals",
    source: "Wikipedia",
    url: "https://en.wikipedia.org/wiki/Equalization_(audio)",
    tag: "พื้นฐาน",
  },
  {
    title: "iZotope Learn (EQ.)",
    source: "iZotope",
    url: "https://www.izotope.com/en/learn.html",
    tag: "พื้นฐาน",
  },
  {
    title: "The EQ Cheat Sheet",
    source: "The EQ",
    url: "https://pandaqi.com/tutorials/audio/music-mixing/the-eq-cheat-sheet/",
    tag: "โทนเสียง",
  },
  {
    title: "Sound On Sound Techniques",
    source: "Sound On Sound",
    url: "https://www.soundonsound.com/techniques",
    tag: "มิกซ์",
  },
  {
    title: "MusicRadar EQ Guides",
    source: "MusicRadar",
    url: "https://www.musicradar.com/search?searchTerm=eq",
    tag: "มิกซ์",
  },
  {
    title: "YouTube: EQ Mixing Tutorials",
    source: "YouTube",
    url: "https://www.youtube.com/results?search_query=eq+mixing+tutorial",
    tag: "มิกซ์",
  },
];

const TAG_STYLE: Record<EqGuideLink["tag"], string> = {
  พื้นฐาน: "border-[#60A5FA]/40 bg-[#1E3A8A]/30 text-[#BFDBFE]",
  มิกซ์: "border-[#A78BFA]/40 bg-[#4C1D95]/30 text-[#DDD6FE]",
  โทนเสียง: "border-[#4ADE80]/40 bg-[#14532D]/30 text-[#BBF7D0]",
};

function ExternalIcon() {
  return (
    <svg className="h-4 w-4 text-[#22D3EE]" viewBox="0 0 20   20" fill="none" aria-hidden="true">
      <path d="M8 12L14 6M10 6H14V10" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
      <path
        d="M14 12V14C14 14.5304 13.7893 15.0391 13.4142 15.4142C13.0391 15.7893 12.5304 16 12 16H6C5.46957 16 4.96086 15.7893 4.58579 15.4142C4.21071 15.0391 4 14.5304 4 14V8C4 7.46957 4.21071 6.96086 4.58579 6.58579C4.96086 6.21071 5.46957 6 6 6H8"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

export default function EqKnowledgePanel() {
  const [isOpen, setIsOpen] = useState(false);
  const panelId = useId();

  return (
    <aside className="fixed left-3 z-40 bottom-3 md:bottom-auto md:top-1/2 md:-translate-y-1/2">
      <nav
        className={`overflow-hidden rounded-2xl border border-[#5B21B6]/35 bg-[#0B1021]/92 shadow-[0_14px_38px_rgba(8,10,28,0.65)] backdrop-blur-lg transition-all duration-300 ${
          isOpen ? "w-[min(88vw,18rem)]" : "w-14"
        }`}
        aria-label="EQ learning links"
      >
        <button
          type="button"
          aria-expanded={isOpen}
          aria-controls={panelId}
          onClick={() => setIsOpen((prev) => !prev)}
          className="flex h-14 w-full items-center gap-2 px-3 text-[#EDE9FE] transition hover:bg-[#111827]"
        >
          <span className="inline-flex h-8 w-8 shrink-0 items-center justify-center rounded-lg border border-[#22D3EE]/45 bg-[#111827]">
            <svg className="h-4 w-4 text-[#22D3EE]" viewBox="0 0 20 20" fill="none" aria-hidden="true">
              <path d="M4 5.75C4 5.33579 4.33579 5 4.75 5H15.25C15.6642 5 16 5.33579 16 5.75V14.25C16 14.6642 15.6642 15 15.25 15H4.75C4.33579 15 4 14.6642 4 14.25V5.75Z" stroke="currentColor" strokeWidth="1.5" />
              <path d="M7 8H13M7 10.5H11M7 13H10.5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
            </svg>
          </span>
          <span
            className={`text-left transition-all duration-200 ${
              isOpen ? "translate-x-0 opacity-100" : "pointer-events-none -translate-x-1 opacity-0"
            }`}
          >
            <span className="block text-xs font-semibold tracking-wide text-[#A78BFA]">EQ GUIDE</span>
            <span className="block text-sm font-bold">เมนูความรู้</span>
          </span>
        </button>

        <div
          id={panelId}
          className={`grid transition-[grid-template-rows,opacity] duration-300 ease-out ${
            isOpen ? "grid-rows-[1fr] opacity-100" : "grid-rows-[0fr] opacity-0"
          }`}
        >
          <div className="min-h-0 overflow-hidden">
            <ul className="max-h-72 space-y-2 overflow-y-auto px-2 pb-2">
              {EQ_GUIDE_LINKS.map((link) => (
                <li key={link.url}>
                  <a
                    href={link.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex min-h-10 items-center justify-between gap-2 rounded-xl border border-[#5B21B6]/30 bg-[#111827]/75 px-3 py-2 text-sm text-[#EDE9FE] transition hover:border-[#22D3EE]/45 hover:bg-[#111827] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#22D3EE]/60"
                  >
                    <span className="min-w-0">
                      <span className="block truncate text-sm font-semibold">{link.title}</span>
                      <span className="mt-0.5 block truncate text-[11px] text-[#9CA3AF]">{link.source}</span>
                    </span>
                    <span className="flex items-center gap-2">
                      <span className={`rounded-md border px-1.5 py-0.5 text-[10px] font-semibold ${TAG_STYLE[link.tag]}`}>{link.tag}</span>
                      <ExternalIcon />
                    </span>
                  </a>
                </li>
              ))}
            </ul>
          </div>
        </div>
      </nav>
    </aside>
  );
}

