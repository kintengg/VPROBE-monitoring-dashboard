"use client"

import Link from "next/link"
import { usePathname, useSearchParams } from "next/navigation"
import { cn } from "@/lib/utils"
import { BantaySidebarLogo } from "@/components/bantay-sidebar-logo"
import { useVideoDomain } from "@/components/video-domain-context"
import {
  Car,
  Cpu,
  Footprints,
  LayoutDashboard,
  ListOrdered,
  User,
} from "lucide-react"

const navItems = [
  { icon: LayoutDashboard, label: "Unified", href: "/" },
  { icon: Footprints, label: "Pedestrian", href: "/pedestrian" },
  { icon: Car, label: "Vehicle", href: "/vehicle" },
  { icon: ListOrdered, label: "Queue", href: "/queue" },
  { icon: Cpu, label: "Models", href: "/models" },
]

export function Sidebar() {
  const pathname = usePathname()
  const videoDomain = useVideoDomain()
  const isVideoRoute = pathname.startsWith("/video")
  const videoRouteMatchesPedestrian = isVideoRoute && videoDomain === "pedestrian"
  const videoRouteMatchesVehicle = isVideoRoute && videoDomain === "vehicle"

  return (
    <aside className="flex h-full w-24 flex-col border-r border-sidebar-border bg-sidebar relative">
      {/* Logo & Branding */}
      <div className="flex h-36 flex-col items-center justify-center gap-2 border-b border-sidebar-border px-2">
        <div className="h-[4.75rem] w-[4.75rem] rounded-[1.75rem] bg-secondary/40 p-1 shadow-elevated-sm ring-1 ring-white/5">
          <BantaySidebarLogo className="h-full w-full" />
        </div>
        <span className="text-[10px] font-extrabold text-white tracking-[0.28em] uppercase">BANTAY</span>
      </div>

      {/* Navigation */}
      <nav className="flex-1 flex flex-col items-center py-6 gap-3">
        {navItems.map((item) => {
          const isVideo = pathname.startsWith("/video")

          const isActive = pathname === item.href ||
            (item.href === "/pedestrian" && (pathname.startsWith("/pedestrian") || pathname === "/search" || videoRouteMatchesPedestrian)) ||
            (item.href === "/queue" && pathname.startsWith("/queue")) ||
            (item.href === "/vehicle" && (pathname.startsWith("/vehicle") || videoRouteMatchesVehicle)) ||
            (item.href === "/models" && pathname.startsWith("/models"))
          return (
            <Link
              key={item.href}
              href={item.href}
              className={cn(
                "flex items-center justify-center w-11 h-11 rounded-2xl transition-all duration-200",
                isActive
                  ? "bg-primary text-primary-foreground shadow-elevated-sm"
                  : "text-muted-foreground hover:text-foreground hover:bg-secondary"
              )}
              title={item.label}
            >
              <item.icon className="w-5 h-5" />
            </Link>
          )
        })}
      </nav>

      {/* Profile Avatar */}
      <div className="flex items-center justify-center pb-6">
        <div className="w-11 h-11 rounded-2xl bg-secondary flex items-center justify-center border border-border">
          <User className="w-5 h-5 text-muted-foreground" />
        </div>
      </div>
    </aside>
  )
}
