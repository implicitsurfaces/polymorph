import lockedIcon from "../assets/input-icons/locked.svg";
import unlockedIcon from "../assets/input-icons/unlocked.svg";

export interface LockIconProps {
	isLocked: boolean;
	onClick: () => void;
}

export function LockIcon({ isLocked, onClick }: LockIconProps) {
	const hint = `Click to ${isLocked ? "unlock" : "lock"}`;
	const icon = isLocked ? lockedIcon : unlockedIcon;
	const colorClass = isLocked ? "red" : "green";
	return (
		<img
			className={`input-icon ${colorClass}`}
			src={icon}
			alt={hint}
			title={hint}
			onClick={onClick}
		/>
	);
}
