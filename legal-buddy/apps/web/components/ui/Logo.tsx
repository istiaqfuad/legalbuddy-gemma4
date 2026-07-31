// Brand mark: an emerald tile with an open-book silhouette — "the law, read and
// cited". Kept to simple filled shapes so it stays crisp from masthead down to a
// 16px favicon. Mirror any change here in app/icon.svg.
export function Logo({ className }: { className?: string }) {
  return (
    <svg
      viewBox="0 0 32 32"
      className={className}
      role="img"
      aria-label="Law Buddy"
      xmlns="http://www.w3.org/2000/svg"
    >
      <defs>
        <linearGradient id="lbLogo" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stopColor="#288A6C" />
          <stop offset="1" stopColor="#185A45" />
        </linearGradient>
      </defs>
      <rect width="32" height="32" rx="7.5" fill="url(#lbLogo)" />
      <g fill="#ffffff">
        <path d="M15 11C11.7 9.9 8.3 9.9 6 10.5L6 21C8.3 20.4 11.7 20.4 15 21.9Z" />
        <path d="M17 11C20.3 9.9 23.7 9.9 26 10.5L26 21C23.7 20.4 20.3 20.4 17 21.9Z" />
      </g>
    </svg>
  );
}
