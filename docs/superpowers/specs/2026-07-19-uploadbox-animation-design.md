# UploadBox Loading & AI Processing Animation Design

## Overview
เป้าหมายของเอกสารนี้คือการออกแบบแอนิเมชัน (Animations) เพื่อยกระดับประสบการณ์ผู้ใช้ (UX) ในช่วงที่ผู้ใช้กำลังรอการประมวลผลจาก AI ใน `UploadBox.tsx` โดยใช้ความสามารถของ **Tailwind CSS 4** ล้วน ๆ เพื่อไม่ให้กระทบประสิทธิภาพของแอปพลิเคชัน

## Components & Visual Changes

### 1. Animated Execute Button (ปุ่มรันคำสั่ง)
**ปัญหาปัจจุบัน:** ใช้ไอคอนวงกลมหมุน (Spinning circle) แบบพื้นฐาน ซึ่งดูเรียบเกินไปสำหรับแอปเพลง
**ดีไซน์ใหม่:**
- สร้างคอมโพเนนต์ไอคอน `<AudioWaveIcon />` ที่ประกอบด้วยแนวตั้ง 3-4 แท่ง
- กำหนด Keyframes (ผ่าน Tailwind หรือ CSS ธรรมดาใน `globals.css`) ให้แต่ละแท่งมีความสูงยืดหดไม่เท่ากันแบบ Random หรือหน่วงเวลาต่างกัน (`animation-delay`)
- เพิ่มแสงรอบปุ่มให้กระเพื่อม (Pulse Glow) ขณะกำลัง Processing

### 2. Segmented Progress Bar (แถบความคืบหน้า)
**ปัญหาปัจจุบัน:** ช่องสี่เหลี่ยมสว่างขึ้นตามเปอร์เซ็นต์ แต่ระหว่างรอเปอร์เซ็นต์ขยับ แถบดูหยุดนิ่ง
**ดีไซน์ใหม่:**
- เพิ่มเอฟเฟกต์แสงวิ่ง (Shimmer/Sweep) ทับไปบนช่องที่สว่างแล้ว (Active segments) 
- ใช้เทคนิค `bg-gradient-to-r` และ `animate-[shimmer_2s_linear_infinite]` เพื่อให้แสงเคลื่อนที่จากซ้ายไปขวาอยู่ตลอดเวลาแม้ขณะติดอยู่ที่ 90% (รอ Backend)

### 3. Skeleton to Real Player Transition (การสลับหน้าจอ)
**ปัญหาปัจจุบัน:** ตอนเปลี่ยนจาก `AudioAnalysisSkeleton` และ `StemMixerSkeleton` ไปเป็น `AudioAnalysis` และ `MultiStemLivePlayer` ตัว UI จะสลับภาพแบบแข็ง ๆ (Snap) ทันที
**ดีไซน์ใหม่:**
- ห่อหุ้มกล่อง Component เหล่านี้ด้วย `div` ที่มีการใช้ `animate-in fade-in slide-in-from-bottom-4 duration-500` (ถ้า Tailwind 4 รองรับ animate-in หรือจะเขียน keyframe `fade-in-up` เอง)
- ผลลัพธ์คือเมื่อโหลดเสร็จ Skeleton จะหายไป และกล่อง Player ของจริงจะค่อยๆ ปรากฏขึ้นพร้อมดันตัวขึ้นมาจากด้านล่างเล็กน้อยอย่างนุ่มนวล

## Architecture & Data Flow
- การเปลี่ยนแปลงนี้เน้นไปที่ UI และ CSS ล้วน ๆ ไม่มีผลกับ Business Logic หรือการยิง API
- ไม่มีการดึง Library ภายนอก (เช่น Framer Motion) มาใช้ เพื่อรักษา Bundle size ให้เบาที่สุดตามเป้าหมาย

## Error Handling
- หากกระบวนการล้มเหลว (เกิด Error) แอนิเมชันทั้งหมดจะหยุดทันที และปุ่ม/แถบสถานะจะเปลี่ยนเป็นกรอบสีแดงพร้อมข้อความแสดงข้อผิดพลาดตามปกติ
