interface PolicyBadgeProps {
  action: string;
  isStop?: boolean;
  size?: 'sm' | 'md';
}

const actionColors: Record<string, string> = {
  RETRY_NOW: 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30',
  RETRY_LATER: 'bg-blue-500/20 text-blue-400 border-blue-500/30',
  REQUEST_UPDATE: 'bg-amber-500/20 text-amber-400 border-amber-500/30',
  HUMAN_REVIEW: 'bg-purple-500/20 text-purple-400 border-purple-500/30',
  STOP: 'bg-red-500/20 text-red-400 border-red-500/30',
  NO_INTERVENTION: 'bg-gray-500/20 text-gray-400 border-gray-500/30',
  INTERVENE: 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30',
};

export default function PolicyBadge({ action, isStop, size = 'md' }: PolicyBadgeProps) {
  const cls = actionColors[action] || 'bg-gray-500/20 text-gray-400 border-gray-500/30';
  const sizeCls = size === 'sm' ? 'text-xs px-2 py-0.5' : 'text-sm px-3 py-1';
  return (
    <span className={`inline-flex items-center rounded border font-medium ${cls} ${sizeCls} ${isStop ? 'ring-1 ring-red-500/50' : ''}`}>
      {action}
    </span>
  );
}
